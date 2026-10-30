# Dataset-v2K — plano versionado da avaliação final

## Estado

Plano `v2K-plan-01` congelado, ainda sem autorização para executar a avaliação. Nenhum membership individual, tensor ou rótulo de teste foi desserializado; não houve inferência nem scoring. v2J1 e v2J2 foram verificados contra seus inventários congelados (19 e 360 arquivos listados, respectivamente) e não foram modificados.

## Alvos congelados

Serão avaliados exatamente 36 checkpoints neurais: nove vencedores A da seleção R4 e 27 controles B/A_outer/B_outer do v2J2, com três seeds por arquitetura e representação. Somam-se os dois artefatos de controle metadata/FOV previstos em v2I: 38 avaliações totais, cada uma sobre as mesmas 73 observações e 61 unidades congeladas. Os 36 checkpoints têm hashes distintos e foram verificados contra os respectivos manifestos.

| Arquitetura | Representação | Seed | Configuração | SHA-256 do checkpoint |
|---|---|---:|---|---|
| custom_cnn_transformer | A | 3908274 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `429327cd27b1788b598b6df0dd55e3098942b299f7b9a5546f00176034983ea6` |
| custom_cnn_transformer | A | 3908275 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `c60e04d53037b29e32f710b31d205d381aaa68e940e7564df2d5822029a75ac7` |
| custom_cnn_transformer | A | 3908276 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `a0c893fcf0d7db3fbffb3f3b7eb31d4362c25b2210dd7b208332d3bbe193407e` |
| custom_cnn_transformer | B | 3908274 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `a59557bb2ce3c18c02897458a6a41d42a2862ae224916b688d34d52d4bdb788a` |
| custom_cnn_transformer | B | 3908275 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `07068bf5ee1ef3ec605dcf1e7a48531f9c9ba2f772d5c03fbaa45efac3e3f742` |
| custom_cnn_transformer | B | 3908276 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `04c61b3346e28abd5e48ccd53c6fddb543c04c0c018b54908e4fb51874053bbb` |
| custom_cnn_transformer | A_outer | 3908274 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `e9a389f8b039e6caa148452ec8a28587e3e52daecc3553a5fc62d825ab1f2a8d` |
| custom_cnn_transformer | A_outer | 3908275 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `b908793c8a567e5dbeefebdcdfd96a39a24f021b8164580e385183342b46613f` |
| custom_cnn_transformer | A_outer | 3908276 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `8b27ac2ae2f9ca0c4308ef3f032773490c010692ec8ccd2b023ba6ced238f2bd` |
| custom_cnn_transformer | B_outer | 3908274 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `57c32f75f034dec8259743926e2c64ad6c7be0cd11a23d665843cd5b42eacc85` |
| custom_cnn_transformer | B_outer | 3908275 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `79761d99eec7ed543fdca0f90e8e0489ebb3f70286817bed35251e1a951b613f` |
| custom_cnn_transformer | B_outer | 3908276 | `spatial-pooling-ablation-v3_spatial_transformer_grid8_mean_grad` | `0e3c4e3a078e9a62a6a5d8410437b79992ad72cb3adf34bf1e0cab5bb3997cde` |
| resnet50 | A | 3908274 | `resnet50_ddc5866488e57f18` | `ab39b0a0c7aad51a40abc79d97e31924d5ca1f9878fc753cf8283a89f2dc624f` |
| resnet50 | A | 3908275 | `resnet50_ddc5866488e57f18` | `3d9e3a31b16c80f83bcfb42bbebd2c37f352369116a0851f360595d18de202a3` |
| resnet50 | A | 3908276 | `resnet50_ddc5866488e57f18` | `4d9f269591e072753cfd57ec32135e249b27f47eb2ad00e067e5fb2a9667f5d5` |
| resnet50 | B | 3908274 | `resnet50_ddc5866488e57f18` | `408273af52a3d25a61e8ecbe0cdbfbeaf10a2047ca558101332004b6d5b10d0f` |
| resnet50 | B | 3908275 | `resnet50_ddc5866488e57f18` | `b70bb71821974be74f95441525505e8f00661d6b4763dbb7a4caf9c3629e1b83` |
| resnet50 | B | 3908276 | `resnet50_ddc5866488e57f18` | `9599578d207f2aae57e73bdf8c81e171e8251186a9dffd1117c65ee9bde8c7ca` |
| resnet50 | A_outer | 3908274 | `resnet50_ddc5866488e57f18` | `70ce94900bab409c97322ad9b38498f087687135149f97aa3d343d1b0d50b0b6` |
| resnet50 | A_outer | 3908275 | `resnet50_ddc5866488e57f18` | `f82129ff96642fb3a001e1f01a59300fd03161b896c67d4e34bbe88f42fe86ba` |
| resnet50 | A_outer | 3908276 | `resnet50_ddc5866488e57f18` | `b08325411b96c421a1d77cd9a0a5d07d2ce080bbcb4866f21aeaefc86d698325` |
| resnet50 | B_outer | 3908274 | `resnet50_ddc5866488e57f18` | `903e08a2787864ad575826b1fa0eec0b508829179cfdb3d7c4479555b0f2c662` |
| resnet50 | B_outer | 3908275 | `resnet50_ddc5866488e57f18` | `eccd23386e4ddb3631024deed224d507b495b1ae60791e2c991758ec39f5c023` |
| resnet50 | B_outer | 3908276 | `resnet50_ddc5866488e57f18` | `729da0d3a3ad8152501904351646f31359fe0a0997bc4a9279423cfc55934dbc` |
| vit_b_16 | A | 3908274 | `vit_b_16_ed7ae4a374295b2c` | `dc27e68c99e5331a2f3e1e001f82825edd5610a62adf2c6db1bfefdcb95003dc` |
| vit_b_16 | A | 3908275 | `vit_b_16_ed7ae4a374295b2c` | `0c46354f19d3aa8a33a81fdbc53a6155cb789da5d4355f953944cfb0539e02c5` |
| vit_b_16 | A | 3908276 | `vit_b_16_ed7ae4a374295b2c` | `73c3892b2a05bc395efbbfd8bc531cd5c8b5073700f5a81583c5982a41cfe88a` |
| vit_b_16 | B | 3908274 | `vit_b_16_ed7ae4a374295b2c` | `4f7b2594c424be8381da6eccc5a10822e0c3f4cf7d875a69eec887786479cff7` |
| vit_b_16 | B | 3908275 | `vit_b_16_ed7ae4a374295b2c` | `da78a5ba4fd8bdcc57bfb466f6a1b596b696e685435c7ab4032455296ce409ab` |
| vit_b_16 | B | 3908276 | `vit_b_16_ed7ae4a374295b2c` | `876e9b6b69a2882276e6ddd703374c5366d8511ef7f5f662607cad1b7aa6bf91` |
| vit_b_16 | A_outer | 3908274 | `vit_b_16_ed7ae4a374295b2c` | `a2f8a5a7c165a569cf6ca6d314e34441fd8b771c11bd4b1d2cd8e47250b94991` |
| vit_b_16 | A_outer | 3908275 | `vit_b_16_ed7ae4a374295b2c` | `c173a3fc42b1820317268c1d2a96f2b470e3f2a5c005910cbd51966cde1d87ab` |
| vit_b_16 | A_outer | 3908276 | `vit_b_16_ed7ae4a374295b2c` | `070bb6ec432fa05239bc3ed9c95c83aaada3e92497f24cc8f2befd3d0e895b49` |
| vit_b_16 | B_outer | 3908274 | `vit_b_16_ed7ae4a374295b2c` | `1d162e823dce9bcc53b122bbc023974d320e2ca64dc9a98b8caa77f1ddcc1412` |
| vit_b_16 | B_outer | 3908275 | `vit_b_16_ed7ae4a374295b2c` | `3158c7d4c1e597282f1142c9a4df4cdd018d6e779601a37989e6d0e887c14ecc` |
| vit_b_16 | B_outer | 3908276 | `vit_b_16_ed7ae4a374295b2c` | `8ab0af9e54eb7856c91dcf14c07aabea41f8ef574ce981dcc929289e7685e6f8` |

Controles metadata/FOV congelados:

| Controle | SHA-256 do artefato |
|---|---|
| metadata_decision_tree_depth3 | `577c2fe802cb0829851113eec95adba702fb49283a6e83be5f4d34a1d93a0a62` |
| metadata_logreg_C1_balanced_lbfgs | `d02226d8ab86d4776f3daf0e6035b2060509ff930d3cd17d04f170609bc18926` |

Manifesto completo com caminhos, tamanhos, configurações e hashes: `evaluation_targets.json`.

## Passagem, isolamento e análise

O plano exige uma única passagem: uma avaliação por alvo × 73 observações (2.628 pares modelo/objeto para redes e 146 para controles metadata). Não há treino, otimização, threshold, seleção por resultado ou repetição de avaliação concluída. A ordem de membership será congelada sem truth; workers recebem somente IDs de objeto/unidade e tensores. Os 38 arquivos de predição, com probabilidades de quatro classes e identidades ordenadas, serão hasheados antes de liberar truth ao scorer isolado.

Serão reportados balanced accuracy, macro-F1 e accuracy por seed; métricas por classe, matrizes de confusão e estratos FOV/extensão; média ± SD amostral (ddof=1) dos três seeds por arquitetura × representação; e os dois controles metadata separadamente. Comparações pareadas, intervalos bootstrap e randomização seguirão literalmente o protocolo v2I: 61 unidades inteiras, PCG64 seed 20261021, 20.000 replicatas compartilhadas e Holm. Não haverá teste t com n=3 seeds nem escolha do melhor seed.

## Integridade e parada

`binding_hashes.json` registra hashes dos protocolos, seleções, manifestos, membership (somente SHA-256 dos bytes), truth manifest (somente SHA-256 dos bytes), checkpoints e código. Antes de qualquer conteúdo de teste, a futura execução terá de congelar e auditar os hashes do executor de inferência e do scorer, hoje ainda inexistentes. Qualquer divergência de identidade, hash, membership, ambiente, preprocessing, probabilidade, selagem ou scoring interrompe a etapa; nenhuma correção ou repetição automática é permitida.

Digest canônico do plano: `c0fd829094c6ad02a9cd4b0099c7accec18743ef576cbedffbce96735941c231`. Digest canônico do manifesto de alvos: `ccb33c9a90b867bf859a1b99db29c87c3b02a8d4cc3fab2c9f48465c775c8942`.
