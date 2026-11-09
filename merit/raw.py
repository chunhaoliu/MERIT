"""Causal raw-replay windows and bounded offline observation references."""
from collections import Counter
import numpy as np
import pandas as pd
from merit.coordinates import runway_positions
from merit.causal import causal_replay


def bounded_reference(times, positions, grid, max_gap):
    """Offline reference only. Exact rows need no future bracket; never extrapolate."""
    times, positions, grid = np.asarray(times), np.asarray(positions), np.asarray(grid)
    if len(times) == 0 or np.any(np.diff(times) < 0):
        raise ValueError('Reference needs nonempty monotone records')
    # Same-time last row, consistent with causal replay.
    unique_last = np.r_[np.flatnonzero(np.diff(times) != 0), len(times)-1]
    t, p = times[unique_last], positions[unique_last]
    left = np.searchsorted(t, grid, side='right')-1
    safe_left = np.clip(left, 0, len(t)-1)
    exact = (left >= 0) & (t[safe_left] == grid)
    right = np.where(exact, safe_left, safe_left+1)
    safe_right = np.clip(right, 0, len(t)-1)
    gap = t[safe_right]-t[safe_left]
    valid = exact | ((left >= 0) & (right < len(t)) & (gap > 0) & (gap <= max_gap))
    fraction = np.divide(grid-t[safe_left], gap, out=np.zeros(len(grid)), where=gap>0)
    result = p[safe_left] + fraction[:,None]*(p[safe_right]-p[safe_left])
    result[~valid] = np.nan
    return {'position_m':result, 'valid':valid,
            'left_index':np.where(valid, unique_last[safe_left], -1),
            'right_index':np.where(valid, unique_last[safe_right], -1)}


def empty_arrays():
    specs = {'history_xy_m':((60,2),float), 'reference_xy_m':((30,2),float),
        'report_age_s':((60,),float), 'changed_position_age_s':((60,),float),
        'timestamp_conflict':((60,),bool), 'input_source_row':((60,),np.int64),
        'input_source_timestamp':((60,),float), 'reference_left_row':((30,),np.int64),
        'reference_right_row':((30,),np.int64), 'origin_timestamp':((),float),
        'aircraft_id':((),str), 'segment_index':((),np.int64)}
    return {k:np.empty((0,*shape),dtype=dtype) for k,(shape,dtype) in specs.items()}


def build_frame(frame, recipe):
    """Original row offsets are zero-based CSV data rows (header excluded)."""
    frame = frame.copy()
    counts = Counter(raw_rows=len(frame), candidate_windows=0, accepted_windows=0,
                     rejected_input_age=0, rejected_reference_gap=0, causal_checks=0)
    frame['_row'] = np.arange(len(frame))
    stamps = pd.to_datetime(frame.Date.astype(str)+' '+frame.Time.astype(str),
                           format='%m/%d/%Y %H:%M:%S.%f',errors='coerce')
    frame['_time'] = (stamps - pd.Timestamp(0)).dt.total_seconds().to_numpy(dtype=float)
    frame.loc[stamps.isna(),'_time'] = np.nan
    for k in ['Range','Bearing','Altitude']:
        frame[k] = pd.to_numeric(frame[k],errors='coerce')
    counts['missing_id_rows'] = int(frame.ID.isna().sum())
    counts['invalid_timestamp_rows'] = int(stamps.isna().sum())
    windows = {k:[] for k in empty_arrays()}
    params = recipe['window']
    if (params['history'],params['future'],params['dt_s']) != (60,30,1):
        raise ValueError('This builder is explicitly limited to the frozen 60/30/1 interface')
    segment_id = 0
    for aircraft, group in frame.groupby('ID',sort=False):
        times = group['_time'].to_numpy()
        if np.any(np.diff(times[np.isfinite(times)]) < 0):
            counts['excluded_reversal_aircraft_days'] += 1
            counts['excluded_reversal_rows'] += len(group)
            continue
        mask = np.isfinite(group[['_time','Range','Bearing','Altitude']]).all(axis=1).to_numpy()
        mask = mask & (group.Range >= 0).to_numpy() & (group.Range < 5).to_numpy() & (group.Altitude < 6000).to_numpy()
        ranks = np.flatnonzero(mask)
        counts['excluded_invalid_or_outside_rows'] += int((~mask).sum())
        if not len(ranks):
            continue
        selected = group.iloc[ranks]
        cuts = np.flatnonzero((np.diff(ranks)>1) |
             (np.diff(selected['_time'].to_numpy()) > params['max_segment_gap_s']))+1
        for indices in np.split(np.arange(len(selected)), cuts):
            segment_id += 1
            counts['segments'] += 1
            seg = selected.iloc[indices]
            t = seg['_time'].to_numpy()
            start, end = int(np.ceil(t[0])), int(np.floor(t[-1]))
            if end-start+1 < 90:
                counts['segments_too_short'] += 1
                continue
            xy = runway_positions(seg.Range.to_numpy(),seg.Bearing.to_numpy())
            rows = seg['_row'].to_numpy()
            grid = np.arange(start,end+1,dtype=float)
            replay = causal_replay(t,xy,grid)
            labels = bounded_reference(t,xy,grid,params['max_label_bracket_gap_s'])
            # Actual-source counterfactual at the first forecast origin per segment.
            origin = grid[59]
            prefix = t <= origin
            truncated = causal_replay(t[prefix],xy[prefix],grid[:60])
            changed = xy.copy()
            changed[~prefix] += 10000.
            perturbed = causal_replay(t,changed,grid[:60])
            for key in replay:
                if not np.array_equal(replay[key][:60],truncated[key],equal_nan=True) or not np.array_equal(replay[key][:60],perturbed[key],equal_nan=True):
                    raise AssertionError(f'Future dependence in {key}')
            counts['causal_checks'] += 1
            for offset in range(0,len(grid)-89,params['stride_s']):
                counts['candidate_windows'] += 1
                x = slice(offset,offset+60)
                y = slice(offset+60,offset+90)
                if not replay['available'][x].all() or (replay['time_since_logged_report_s'][x] > params['max_input_report_age_s']).any():
                    counts['rejected_input_age'] += 1
                    continue
                if not labels['valid'][y].all():
                    counts['rejected_reference_gap'] += 1
                    continue
                ix = replay['source_row_index'][x]
                item = {'history_xy_m':replay['position_m'][x], 'reference_xy_m':labels['position_m'][y],
                    'report_age_s':replay['time_since_logged_report_s'][x],
                    'changed_position_age_s':replay['time_since_changed_position_s'][x],
                    'timestamp_conflict':replay['same_timestamp_position_conflict'][x],
                    'input_source_row':rows[ix], 'input_source_timestamp':t[ix],
                    'reference_left_row':rows[labels['left_index'][y]],
                    'reference_right_row':rows[labels['right_index'][y]],
                    'origin_timestamp':grid[offset+59], 'aircraft_id':str(aircraft),
                    'segment_index':segment_id}
                for k,v in item.items():
                    windows[k].append(v)
                counts['accepted_windows'] += 1
    arrays = {k:np.asarray(windows[k]) if windows[k] else v for k,v in empty_arrays().items()}
    assert counts['candidate_windows'] == counts['accepted_windows']+counts['rejected_input_age']+counts['rejected_reference_gap']
    if counts['accepted_windows']:
        grid = arrays['origin_timestamp'][:,None]+np.arange(-59,1)
        assert np.all(arrays['input_source_timestamp'] <= grid)
        assert np.isfinite(arrays['history_xy_m']).all() and np.isfinite(arrays['reference_xy_m']).all()
    return arrays, dict(counts)
