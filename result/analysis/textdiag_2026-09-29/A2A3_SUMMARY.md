# A1/A2/A3 on exploratory anchor runs (Gumbel ON; own N), 3 seeds, first 500 validation rows, CPU deployment forward

## A2 code->own-axis (chance .25) and geometry

| dataset | arm | n | pre-quant token | codeword | eff-rank z (mean over slots) | cos(z,q) | codewords used / K | proto~codeword cos |
|---|---|---:|---:|---:|---:|---:|---|---:|
| CIFAR10 | anchors | 3 | 0.273 ± 0.030 | 0.332 ± 0.051 | 16.119 ± 0.262 | 0.694 ± 0.033 | 42.2 / 64 | 0.796 ± 0.020 |
| Flickr25k | anchors | 3 | 0.273 ± 0.017 | 0.370 ± 0.014 | 46.726 ± 2.033 | 0.661 ± 0.006 | 81.0 / 128 | 0.797 ± 0.007 |
| Flickr25k | base | 3 | 0.267 ± 0.003 | 0.307 ± 0.007 | 32.833 ± 1.099 | 0.708 ± 0.005 | 76.8 / 128 | 0.819 ± 0.007 |
| Flickr25k | notext | 3 | 0.249 ± 0.004 | 0.252 ± 0.004 | 45.473 ± 0.924 | 0.684 ± 0.009 | 85.4 / 128 | 0.869 ± 0.012 |
| MSCOCO | anchors | 3 | 0.260 ± 0.010 | 0.418 ± 0.023 | 71.141 ± 1.368 | 0.657 ± 0.005 | 108.9 / 128 | 0.821 ± 0.006 |
| NUSWIDE | anchors | 3 | 0.293 ± 0.011 | 0.444 ± 0.018 | 43.412 ± 0.435 | 0.708 ± 0.004 | 110.3 / 128 | 0.826 ± 0.002 |

## A1 the text_code_kl target on the validation rows (mean over the 4 local slots)

| dataset | arm | mean confidence | share excluded (<= 0.2) | text argmax = visual codeword | text top-1 mass |
|---|---|---:|---:|---:|---:|
| CIFAR10 | anchors | 0.526 ± 0.009 | 0.006 ± 0.003 | 0.093 ± 0.018 | 0.351 ± 0.012 |
| Flickr25k | anchors | 0.314 ± 0.017 | 0.139 ± 0.029 | 0.098 ± 0.010 | 0.181 ± 0.012 |
| Flickr25k | base | 0.414 ± 0.010 | 0.014 ± 0.002 | 0.152 ± 0.004 | 0.260 ± 0.002 |
| Flickr25k | notext | 0.026 ± 0.001 | 1.000 ± 0.000 | 0.011 ± 0.001 | 0.024 ± 0.000 |
| MSCOCO | anchors | 0.359 ± 0.005 | 0.079 ± 0.008 | 0.116 ± 0.004 | 0.233 ± 0.006 |
| NUSWIDE | anchors | 0.377 ± 0.008 | 0.026 ± 0.005 | 0.106 ± 0.009 | 0.211 ± 0.007 |

## A3 cross-image slot consistency, pairing rule: caption word Jaccard >= 0.25

P(same codeword) lift over all pairs: OWN slot vs the OTHER three slots (same pairs). Base-Hamming delta of the own-slot codon vs all pairs.

| dataset | arm | axis | pairs | own-slot lift | other-slots lift | own − other | Δ base-Hamming (own) |
|---|---|---|---:|---:|---:|---:|---:|
| CIFAR10 | anchors | primary_object | 1563 | 8.139 ± 1.366 | 8.293 ± 0.810 | -0.154 ± 1.796 | -1.175 ± 0.243 |
| CIFAR10 | anchors | secondary_object | 1078 | 3.683 ± 0.126 | 4.224 ± 0.196 | -0.542 ± 0.101 | -0.616 ± 0.168 |
| CIFAR10 | anchors | activity_or_relation | 3922 | 4.431 ± 1.188 | 3.888 ± 0.426 | 0.543 ± 1.128 | -0.816 ± 0.160 |
| CIFAR10 | anchors | color_texture | 4385 | 4.151 ± 0.315 | 4.123 ± 0.159 | 0.028 ± 0.208 | -0.775 ± 0.151 |
| Flickr25k | anchors | primary_object | 136 | 9.026 ± 1.755 | 8.938 ± 1.125 | 0.087 ± 0.724 | -0.997 ± 0.008 |
| Flickr25k | anchors | secondary_object | 75 | 6.532 ± 1.156 | 7.615 ± 0.283 | -1.083 ± 1.428 | -0.576 ± 0.049 |
| Flickr25k | anchors | activity_or_relation | 66 | 9.699 ± 0.066 | 7.465 ± 0.818 | 2.234 ± 0.807 | -0.700 ± 0.042 |
| Flickr25k | anchors | color_texture | 880 | 4.882 ± 0.809 | 4.763 ± 0.439 | 0.119 ± 1.190 | -0.425 ± 0.051 |
| Flickr25k | base | primary_object | 136 | 7.781 ± 3.749 | 7.325 ± 0.889 | 0.456 ± 4.077 | -0.805 ± 0.061 |
| Flickr25k | base | secondary_object | 75 | 5.690 ± 3.777 | 7.374 ± 0.893 | -1.684 ± 4.669 | -0.451 ± 0.141 |
| Flickr25k | base | activity_or_relation | 66 | 8.174 ± 5.625 | 8.823 ± 2.325 | -0.648 ± 6.678 | -0.710 ± 0.209 |
| Flickr25k | base | color_texture | 880 | 4.690 ± 0.196 | 4.510 ± 0.268 | 0.180 ± 0.127 | -0.481 ± 0.042 |
| Flickr25k | notext | primary_object | 136 | 10.460 ± 4.408 | 9.551 ± 1.690 | 0.910 ± 2.730 | -0.797 ± 0.201 |
| Flickr25k | notext | secondary_object | 75 | 9.003 ± 2.052 | 7.764 ± 0.787 | 1.239 ± 1.268 | -0.603 ± 0.045 |
| Flickr25k | notext | activity_or_relation | 66 | 9.706 ± 4.685 | 9.137 ± 1.756 | 0.569 ± 2.936 | -0.636 ± 0.020 |
| Flickr25k | notext | color_texture | 880 | 4.896 ± 0.048 | 4.906 ± 0.442 | -0.010 ± 0.469 | -0.403 ± 0.062 |
| MSCOCO | anchors | primary_object | 841 | 8.595 ± 1.790 | 7.910 ± 1.441 | 0.685 ± 2.239 | -0.869 ± 0.064 |
| MSCOCO | anchors | secondary_object | 139 | 13.698 ± 2.818 | 11.382 ± 3.073 | 2.316 ± 5.857 | -1.213 ± 0.056 |
| MSCOCO | anchors | activity_or_relation | 285 | 6.451 ± 1.895 | 6.322 ± 0.994 | 0.129 ± 1.672 | -0.855 ± 0.033 |
| MSCOCO | anchors | color_texture | 2009 | 3.777 ± 1.090 | 3.522 ± 0.281 | 0.256 ± 1.050 | -0.441 ± 0.034 |
| NUSWIDE | anchors | primary_object | 182 | 14.627 ± 2.040 | 11.584 ± 1.916 | 3.043 ± 3.765 | -1.030 ± 0.048 |
| NUSWIDE | anchors | secondary_object | 265 | 5.647 ± 2.647 | 6.384 ± 1.103 | -0.737 ± 1.545 | -0.635 ± 0.109 |
| NUSWIDE | anchors | activity_or_relation | 161 | 9.897 ± 0.900 | 12.364 ± 0.565 | -2.466 ± 1.404 | -0.833 ± 0.071 |
| NUSWIDE | anchors | color_texture | 2526 | 4.775 ± 0.306 | 4.590 ± 0.133 | 0.185 ± 0.433 | -0.407 ± 0.079 |

## A3 cross-image slot consistency, pairing rule: caption CLIP cosine top 2 %

P(same codeword) lift over all pairs: OWN slot vs the OTHER three slots (same pairs). Base-Hamming delta of the own-slot codon vs all pairs.

| dataset | arm | axis | pairs | own-slot lift | other-slots lift | own − other | Δ base-Hamming (own) |
|---|---|---|---:|---:|---:|---:|---:|
| CIFAR10 | anchors | primary_object | 2495 | 6.837 ± 1.773 | 6.965 ± 0.929 | -0.128 ± 2.376 | -0.977 ± 0.250 |
| CIFAR10 | anchors | secondary_object | 2495 | 3.644 ± 0.072 | 3.947 ± 0.261 | -0.303 ± 0.334 | -0.527 ± 0.098 |
| CIFAR10 | anchors | activity_or_relation | 2495 | 7.932 ± 1.929 | 7.196 ± 0.822 | 0.736 ± 1.770 | -1.289 ± 0.042 |
| CIFAR10 | anchors | color_texture | 2495 | 4.159 ± 0.508 | 4.221 ± 0.358 | -0.061 ± 0.150 | -0.806 ± 0.138 |
| Flickr25k | anchors | primary_object | 2495 | 4.769 ± 0.933 | 5.127 ± 0.298 | -0.358 ± 0.759 | -0.506 ± 0.089 |
| Flickr25k | anchors | secondary_object | 2495 | 3.272 ± 0.115 | 3.495 ± 0.186 | -0.223 ± 0.111 | -0.340 ± 0.045 |
| Flickr25k | anchors | activity_or_relation | 2495 | 6.107 ± 0.033 | 4.868 ± 0.593 | 1.239 ± 0.591 | -0.584 ± 0.041 |
| Flickr25k | anchors | color_texture | 2495 | 2.517 ± 0.338 | 2.570 ± 0.305 | -0.052 ± 0.235 | -0.174 ± 0.023 |
| Flickr25k | base | primary_object | 2495 | 4.708 ± 1.447 | 5.161 ± 0.936 | -0.453 ± 2.252 | -0.514 ± 0.076 |
| Flickr25k | base | secondary_object | 2495 | 3.301 ± 1.588 | 3.689 ± 0.407 | -0.388 ± 1.883 | -0.319 ± 0.057 |
| Flickr25k | base | activity_or_relation | 2495 | 5.032 ± 1.313 | 4.656 ± 0.783 | 0.376 ± 2.047 | -0.478 ± 0.100 |
| Flickr25k | base | color_texture | 2495 | 2.690 ± 0.429 | 2.633 ± 0.164 | 0.057 ± 0.271 | -0.241 ± 0.010 |
| Flickr25k | notext | primary_object | 2495 | 5.531 ± 0.924 | 5.284 ± 0.384 | 0.247 ± 0.876 | -0.463 ± 0.032 |
| Flickr25k | notext | secondary_object | 2495 | 3.384 ± 0.069 | 3.573 ± 0.159 | -0.189 ± 0.226 | -0.281 ± 0.061 |
| Flickr25k | notext | activity_or_relation | 2495 | 5.169 ± 0.342 | 5.223 ± 0.369 | -0.054 ± 0.394 | -0.476 ± 0.074 |
| Flickr25k | notext | color_texture | 2495 | 2.552 ± 0.359 | 2.665 ± 0.095 | -0.113 ± 0.264 | -0.162 ± 0.027 |
| MSCOCO | anchors | primary_object | 2495 | 5.254 ± 1.552 | 5.262 ± 0.944 | -0.008 ± 1.343 | -0.734 ± 0.024 |
| MSCOCO | anchors | secondary_object | 2495 | 4.200 ± 0.906 | 3.690 ± 0.691 | 0.510 ± 1.561 | -0.512 ± 0.052 |
| MSCOCO | anchors | activity_or_relation | 2495 | 3.469 ± 0.699 | 3.264 ± 0.247 | 0.205 ± 0.489 | -0.435 ± 0.070 |
| MSCOCO | anchors | color_texture | 2495 | 2.705 ± 0.717 | 2.675 ± 0.256 | 0.031 ± 0.558 | -0.296 ± 0.025 |
| NUSWIDE | anchors | primary_object | 2495 | 7.636 ± 0.649 | 6.776 ± 0.703 | 0.859 ± 1.349 | -0.675 ± 0.039 |
| NUSWIDE | anchors | secondary_object | 2495 | 4.896 ± 1.120 | 5.360 ± 0.162 | -0.464 ± 0.959 | -0.577 ± 0.094 |
| NUSWIDE | anchors | activity_or_relation | 2495 | 7.128 ± 1.282 | 6.986 ± 0.269 | 0.142 ± 1.544 | -0.634 ± 0.024 |
| NUSWIDE | anchors | color_texture | 2495 | 3.815 ± 0.459 | 4.193 ± 0.570 | -0.378 ± 0.906 | -0.378 ± 0.040 |
