# Dataset-v2J2 — controles de representação

## Execução

Foram concluídos e verificados 27/27 fits sequenciais: 3 arquiteturas × 3 representações (B, A_outer, B_outer) × 3 seeds. GPU acumulada: 0.840 h; wall da campanha: 2.942 h. Falhas: 0; retomadas: 0. Cada registro de conclusão e seus artefatos, incluindo best/last checkpoints, histórico e predições de validação, tiveram os hashes conferidos.

Protocolo v2I: `6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737`. Seleção R4 vinculada: `cd45b0c9fbb1e6310b31410c5f3be5edf9e694a7737f17a6c95c0788381a580b`. Cada fit usou a configuração vencedora R4 da arquitetura correspondente, sem seleção baseada nos controles.

## Métricas de validação por fit

Estes valores são resultados de validação usados pelo protocolo de treino/checkpoint; não são métricas de teste. Nenhum controle foi escolhido ou descartado pelo desempenho.

| Família | Representação | Seed | Accuracy | Balanced accuracy | Macro-F1 | Weighted loss | Época best | GPU s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| custom_cnn_transformer | B | 3908274 | 0.693333 | 0.691176 | 0.676125 | 0.842147 | 31 | 180.8 |
| custom_cnn_transformer | B | 3908275 | 0.666667 | 0.666176 | 0.656257 | 0.891281 | 32 | 176.4 |
| custom_cnn_transformer | B | 3908276 | 0.706667 | 0.706540 | 0.695413 | 0.856306 | 30 | 170.2 |
| custom_cnn_transformer | A_outer | 3908274 | 0.720000 | 0.702012 | 0.663969 | 0.781574 | 23 | 146.8 |
| custom_cnn_transformer | A_outer | 3908275 | 0.760000 | 0.748994 | 0.739538 | 0.710033 | 32 | 177.8 |
| custom_cnn_transformer | A_outer | 3908276 | 0.746667 | 0.732972 | 0.718726 | 0.736743 | 26 | 157.6 |
| custom_cnn_transformer | B_outer | 3908274 | 0.626667 | 0.618073 | 0.612520 | 1.051463 | 27 | 155.5 |
| custom_cnn_transformer | B_outer | 3908275 | 0.586667 | 0.577283 | 0.563345 | 1.048697 | 22 | 136.4 |
| custom_cnn_transformer | B_outer | 3908276 | 0.573333 | 0.562577 | 0.549021 | 1.056441 | 15 | 115.1 |
| resnet50 | B | 3908274 | 0.773333 | 0.768344 | 0.768105 | 0.779838 | 6 | 54.2 |
| resnet50 | B | 3908275 | 0.720000 | 0.715054 | 0.712033 | 0.791607 | 4 | 50.4 |
| resnet50 | B | 3908276 | 0.680000 | 0.676006 | 0.677252 | 0.920044 | 4 | 49.9 |
| resnet50 | A_outer | 3908274 | 0.626667 | 0.623336 | 0.614882 | 0.965448 | 2 | 46.6 |
| resnet50 | A_outer | 3908275 | 0.680000 | 0.681772 | 0.662840 | 0.825357 | 1 | 44.4 |
| resnet50 | A_outer | 3908276 | 0.840000 | 0.831269 | 0.830837 | 0.823601 | 11 | 65.7 |
| resnet50 | B_outer | 3908274 | 0.573333 | 0.569195 | 0.569577 | 1.228401 | 4 | 48.5 |
| resnet50 | B_outer | 3908275 | 0.733333 | 0.723568 | 0.718714 | 1.132455 | 6 | 52.4 |
| resnet50 | B_outer | 3908276 | 0.693333 | 0.688506 | 0.687698 | 1.044676 | 14 | 69.4 |
| vit_b_16 | B | 3908274 | 0.733333 | 0.729102 | 0.722691 | 0.791052 | 5 | 120.6 |
| vit_b_16 | B | 3908275 | 0.786667 | 0.785488 | 0.781034 | 0.684823 | 3 | 109.7 |
| vit_b_16 | B | 3908276 | 0.800000 | 0.799536 | 0.793780 | 0.816034 | 4 | 115.3 |
| vit_b_16 | A_outer | 3908274 | 0.760000 | 0.753870 | 0.749718 | 0.695180 | 3 | 116.6 |
| vit_b_16 | A_outer | 3908275 | 0.773333 | 0.768344 | 0.768243 | 0.651531 | 4 | 119.1 |
| vit_b_16 | A_outer | 3908276 | 0.800000 | 0.791796 | 0.789758 | 0.655475 | 3 | 115.8 |
| vit_b_16 | B_outer | 3908274 | 0.720000 | 0.717028 | 0.718686 | 0.948448 | 9 | 148.5 |
| vit_b_16 | B_outer | 3908275 | 0.653333 | 0.655650 | 0.655448 | 0.888049 | 8 | 143.3 |
| vit_b_16 | B_outer | 3908276 | 0.733333 | 0.726432 | 0.724530 | 0.837194 | 7 | 138.1 |

## Isolamento e escopo

Os workers receberam somente treino e validação. Nenhum tensor ou rótulo de teste foi aberto; não houve inferência, scoring, bootstrap, randomização ou avaliação final no teste. Não houve retreino de A, busca adicional, reseleção, alteração de configuração ou avaliação de teste. A avaliação final permanece bloqueada e requer autorização separada.

O manifesto dos 27 checkpoints, contabilidade, métricas e validação estão nos artefatos versionados desta fase.
