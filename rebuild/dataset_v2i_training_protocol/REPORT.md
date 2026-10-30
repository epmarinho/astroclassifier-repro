# Dataset-v2I — protocolo congelado após decisões autorais

## Estado

**Pronto para execução: `ready_for_execution=true`.** As cinco decisões foram congeladas prospectivamente. Esta fase não executou treino: nenhum checkpoint foi carregado, nenhuma inferência ou pontuação ocorreu e nenhum rótulo de teste foi desserializado. Acesso aos rótulos continua bloqueado até previsões finais serem seladas.

## Plano e contagens

Adotada Alternative B: **90 fits de busca A/validação** (36 ResNet50, 36 ViT-B/16 e 18 Custom CNN+Transformer), seguidos por **27 fits de controle** (B, A_outer e B_outer × três seeds para cada vencedor) e **dois fits fixos metadata/FOV**. Total: **119 fits** (117 rede + 2 metadata). Os nove fits A dos configs vencedores são reutilizados; não há repetição. A seleção usa somente a regra de validação congelada, escolhe uma configuração por arquitetura e nunca escolhe o melhor seed. A avaliação final planejada é uma passagem por 36 checkpoints de rede e dois controles metadata: 38 passagens, uma vez, depois do congelamento da seleção.

## Adaptador do custom

Mantida a arquitetura efetiva v1 e sua entrada 256×256. Para cada representação A, B, A_outer e B_outer, após carregar o tensor v2H e aplicar apenas a augmentação D4 de treino congelada (identidade em validação/teste), aplicar em CPU, float32, sem parâmetros treináveis: `torch.nn.functional.interpolate(x.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False, antialias=True).squeeze(0).contiguous()`. Ambiente congelado: Python 3.12.3, torch 2.7.0+cu128, torchvision 0.22.0+cu128. Em seguida transferir para CUDA e chamar o modelo. Arquivos v2H não são modificados; adapter não é representação nem candidato e a arquitetura não muda para grid 7.

## Transferência e orçamento

ResNet50 e ViT-B/16 usam exclusivamente os pesos oficiais locais com SHA-256 `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a` e `c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9`; ausência ou diferença interrompe a execução. Entrada blue/red/NIR nessa ordem, com normalização v2H exata, sem normalização ImageNet/familiar adicional. Os canais não são RGB convencional; trata-se de política congelada de transferência, sem alegação de equivalência espectral.

Estimativa aceita: **16,5 GPU-h centrais**, faixa indicativa **10,9–40,8 h**. A estimativa custom é proxy de planejamento, não runtime v2 medido. Nenhum piloto foi executado e a contagem de candidatos não foi alterada.

## Estatística e lock

Usar 61 unidades de atribuição indivisíveis, NumPy PCG64 seed 20261021, 20.000 replicatas de bootstrap cluster pareado e 20.000 replicatas de randomização pareada. Todos os objetos de uma unidade ficam juntos; as mesmas amostras servem a modelos, representações, métricas e seeds. Workers não recebem truth; selar as previsões antes de entregar truth ao scorer. Resultados não podem disparar treino, substituição ou reseleção.

## Preservação e digest

Foram verificados novamente os hashes de v2A–v2H: v2A–v2G **9.209 arquivos** e v2H **1.496 arquivos**, sem alteração. Arrays não foram reconstruídos. O manifesto de labels v2H foi verificado somente por hash, sem leitura/desserialização.

Digest canônico do protocolo: `6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737`. Estimativa e limites operacionais, regras de resume/falha, hiperparâmetros, ambiente e fontes permanecem definidos nos artefatos desta fase.
