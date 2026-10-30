# Dataset-v2J1-R4 — seleção congelada a partir dos recibos R3

Protocolo: `6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737`. Digest da seleção R4: `cd45b0c9fbb1e6310b31410c5f3be5edf9e694a7737f17a6c95c0788381a580b`.

O defeito R3 era de esquema: `candidate_configurations_by_family` contém somente `configuration_id` e `configuration_sha256`. Os parâmetros efetivos estão em `configuration.json → slot → effective_parameters`, repetidos de forma idêntica nos três seeds de cada configuração e vinculados pelo hash canônico do slot. R4 usa esse campo e as métricas já registradas em `fit_complete.json`; não leu previsões nem recalculou métricas.

Foram validados **90/90** fits de busca e **2/2** controles metadata/FOV. Duas recomputações independentes produziram bytes idênticos; SHA-256: `52643419e9ad8a15480ee821baf19bf655d6d9e62c374386b544b8f38fe236a2`. A seleção canônica tem digest `cd45b0c9fbb1e6310b31410c5f3be5edf9e694a7737f17a6c95c0788381a580b`.

A regra congelada escolhe uma configuração por arquitetura por: maior média de balanced accuracy nas três seeds; maior média de macro-F1; menor média de weighted validation loss; menor variância amostral da balanced accuracy (`ddof=1`); e menor identificador lexical de configuração. O identificador disponível no manifesto é `configuration_id`, usado pelo executor R3 como chave de desempate. Nenhuma seed individual ou controle metadata/FOV influenciou a seleção.

## custom_cnn_transformer

Vencedora: `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad`. Média BA `0.76418989`, macro-F1 `0.75239461`, loss `0.66279796`, variância BA `0.0002452178525 `.

| Seed | Fit ID | BA | Macro-F1 | Loss | Checkpoint SHA-256 |
|---:|---|---:|---:|---:|---|
| 3908274 | fit_c05073a85f17c6b4c0e378f2 | 0.74613003 | 0.73309818 | 0.68069091 | `429327cd27b1788b598b6df0dd55e3098942b299f7b9a5546f00176034983ea6` |
| 3908275 | fit_46c77751cd744157c900ce49 | 0.77244582 | 0.75759722 | 0.63678636 | `c60e04d53037b29e32f710b31d205d381aaa68e940e7564df2d5822029a75ac7` |
| 3908276 | fit_928e6367d59d3fe75b659283 | 0.77399381 | 0.76648841 | 0.67091661 | `a0c893fcf0d7db3fbffb3f3b7eb31d4362c25b2210dd7b208332d3bbe193407e` |

## resnet50

Vencedora: `resnet50_ddc5866488e57f18`. Média BA `0.85961042`, macro-F1 `0.85945023`, loss `0.56295773`, variância BA `2.617371089e-05 `.

| Seed | Fit ID | BA | Macro-F1 | Loss | Checkpoint SHA-256 |
|---:|---|---:|---:|---:|---|
| 3908274 | fit_d807ccc78f9f55959d821058 | 0.86288700 | 0.86430921 | 0.50930950 | `ab39b0a0c7aad51a40abc79d97e31924d5ca1f9878fc753cf8283a89f2dc624f` |
| 3908275 | fit_8ea7d46d25e3ff4711aa872c | 0.86222910 | 0.86383835 | 0.62025631 | `3d9e3a31b16c80f83bcfb42bbebd2c37f352369116a0851f360595d18de202a3` |
| 3908276 | fit_213b98f5b2335e6839c34835 | 0.85371517 | 0.85020314 | 0.55930740 | `4d9f269591e072753cfd57ec32135e249b27f47eb2ad00e067e5fb2a9667f5d5` |

## vit_b_16

Vencedora: `vit_b_16_ed7ae4a374295b2c`. Média BA `0.85794634`, macro-F1 `0.85553697`, loss `0.43492245`, variância BA `0.002005938446 `.

| Seed | Fit ID | BA | Macro-F1 | Loss | Checkpoint SHA-256 |
|---:|---|---:|---:|---:|---|
| 3908274 | fit_22fc888eff7d0577f8d02dc9 | 0.82341331 | 0.82341331 | 0.40078599 | `dc27e68c99e5331a2f3e1e001f82825edd5610a62adf2c6db1bfefdcb95003dc` |
| 3908275 | fit_3a83dde1ef421ea30de958c6 | 0.84187307 | 0.83799794 | 0.52453473 | `0c46354f19d3aa8a33a81fdbc53a6155cb789da5d4355f953944cfb0539e02c5` |
| 3908276 | fit_d7801df3f99349df3403e22a | 0.90855263 | 0.90519965 | 0.37944665 | `73c3892b2a05bc395efbbfd8bc531cd5c8b5073700f5a81583c5982a41cfe88a` |

## Controles e preservação

Os dois controles metadata/FOV estão completos e seus resultados permanecem fora da regra de seleção. Attempt_01 e attempt_02 continuam registrados como falhas operacionais. As 90 fits do attempt_03 permanecem intactas. O `selection.json` da raiz foi copiado byte a byte para `selection_pre_R3.json`; ele registra o estado anterior à campanha R3, não é entrada nem resultado desta seleção.

Nenhum treino, forward pass, inferência, novo scoring, acesso a tensores/rótulos de treino ou teste, ou v2J2 ocorreu nesta fase. Não houve attempt_04. O recomputador R3, o log da falha e os resultados R3 foram preservados e hash verificados.

Artefatos de seleção e hashes completos estão em `frozen_selection_r4.json`, `selected_checkpoint_manifest_r4.json`, `per_fit_validation_metrics_r4.csv` e `r4_provenance.json`.
