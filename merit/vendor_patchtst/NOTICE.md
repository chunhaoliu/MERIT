# Official PatchTST dependency attribution

Source: https://github.com/yuqinie98/PatchTST, supervised implementation,
commit204c21efe0b39603ad6e2ca640ef5896646ab1a9 (ICLR2023).
Apache2 license is retained in LICENSE. RevIN.py retains its original attribution
to https://github.com/ts-kim/RevIN.

PatchTST_layers.py and RevIN.py are byte-identical to upstream. Only the two
backbone absolute imports were changed to package-relative imports. No algorithm
or layer changes. provenance.json records original and local SHA256 hashes.
Task-specific prefix reconstruction and training live outside this dependency.
