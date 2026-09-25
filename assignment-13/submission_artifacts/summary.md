| run | variant | batch | tokens / steps | peak lr | train loss | val loss | val ppl | tokens/s | step | peak memory | activations saved/step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 · baseline, fixed batch | `none` | 32 × 256 | 5,005,312 / 611 | 1.0e-03 | 2.7878 | **2.6964** | 14.8 | **1,869** | 4.37 s | **4.15 GiB** | 2,579 MiB |
| 2 · reversible, fixed batch | `momentum` | 32 × 256 | 5,005,312 / 611 | 1.0e-03 | 2.7094 | **2.6332** | 13.9 | **1,345** | 6.07 s | **2.67 GiB** | 560 MiB |
| 3 · reversible, max batch | `momentum` | 312 × 256 | 5,031,936 / 63 | 3.0e-03 | 5.0508 | **4.9749** | 144.7 | **986** | 80.97 s | **10.95 GiB** | 5,462 MiB |
