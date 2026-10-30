# Dataset-v2J1 — busca e seleção congeladas

A busca v2J1 concluiu **90/90 fits científicos** em representação A e **2/2 controles metadata/FOV**. A seleção foi congelada na fase versionada R4, usando somente métricas salvas nos recibos de conclusão dos fits. Digest da seleção: `cd45b0c9fbb1e6310b31410c5f3be5edf9e694a7737f17a6c95c0788381a580b`. Protocolo: `6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737`.

## Seleção por arquitetura

- `custom_cnn_transformer`: `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad`, média BA 0.764190, macro-F1 0.752395, loss 0.662798. Três checkpoints e hashes estão em `repair_r4/selected_checkpoint_manifest_r4.json`.
- `resnet50`: `resnet50_ddc5866488e57f18`, média BA 0.859610, macro-F1 0.859450, loss 0.562958. Três checkpoints e hashes estão em `repair_r4/selected_checkpoint_manifest_r4.json`.
- `vit_b_16`: `vit_b_16_ed7ae4a374295b2c`, média BA 0.857946, macro-F1 0.855537, loss 0.434922. Três checkpoints e hashes estão em `repair_r4/selected_checkpoint_manifest_r4.json`.

## Proveniência e limites

R3 parou no recomputador por `KeyError: effective_parameters`, depois de os 90 fits e os dois controles concluírem. A falha foi preservada. R4 identificou os parâmetros em `configuration.json → slot → effective_parameters`; os resultados de cada seed vieram de `fit_complete.json → validation_metrics`. Nenhuma métrica foi calculada novamente. Duas recomputações independentes foram byte a byte idênticas.

A regra de seleção usou a média de balanced accuracy nas três seeds; desempates por média de macro-F1, menor loss média, menor variância amostral (`ddof=1`) da BA e menor `configuration_id` lexical. As configurações e checkpoints estão congelados para a seleção A; controles não influenciaram a escolha.

O `selection.json` da raiz foi preservado sem alteração e é explicitamente o artefato anterior à campanha R3. A seleção válida desta campanha está em `repair_r4/frozen_selection_r4.json`.

Nenhum tensor ou rótulo de treino/teste foi aberto nesta fase; não houve treino, forward pass, inferência, novo scoring, v2J2 ou attempt_04. A avaliação final de teste não foi executada.

Relatório detalhado, diff R3→R4, validação, proveniência e inventário estão em `repair_r4/`.
