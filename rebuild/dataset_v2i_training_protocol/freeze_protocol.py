#!/usr/bin/env python3
"""Write the prospective Dataset-v2I protocol and author gates; no execution."""
import hashlib,json
from pathlib import Path
R=Path('rebuild/dataset_v2i_training_protocol')
F=Path('rebuild/dataset_v2f_representation_split_protocol')
G=Path('rebuild/dataset_v2g_split_materialization')
H=Path('rebuild/dataset_v2h_representation_materialization')
V1=Path('rebuild/baseline_comparison_v1')
P4C=V1/'phase4c_selection_freeze'; P4B=V1/'phase4b/attempt_4b1a';P4E=V1/'phase4e_object_independence'
def canon(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def dump(n,x):(R/n).write_text(json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')

stats='''# Plano estatístico prospectivo — Dataset-v2I

## Unidade e métricas

A unidade de inferência é a unidade de atribuição/componente congelada em v2G. O teste tem 73 objetos agrupados em 61 unidades. Cada objeto mantém sua previsão e probabilidades; objetos da mesma unidade não são tratados como réplicas independentes. As métricas pontuais convencionais são calculadas sobre os objetos e são descritivas. Intervalos e testes reamostram unidades completas, mantendo juntos todos os objetos de cada componente.

Para cada seed, arquitetura e representação, relatar balanced accuracy, macro-F1 e accuracy, além de precisão, recall, F1 e suporte por classe e matriz de confusão. Relatar todos os seeds e média ± desvio-padrão amostral (ddof=1); não escolher o melhor seed.

Balanced accuracy é a média do recall nas classes com suporte verdadeiro positivo; macro-F1 inclui as quatro classes fixas, usando zero quando o denominador é zero. Reportar suporte zero, métricas indefinidas e replicatas degeneradas sem suprimi-las.

## Bootstrap pareado

Usar NumPy PCG64, 20.000 replicatas, seed 20261021, intervalos percentis bilaterais de 95% com quantil linear. Cada replicata amostra 61 unidades com reposição; todos os objetos da unidade sorteada entram juntos. O mesmo sorteio é usado para todos os modelos, representações, métricas e seeds. Calcular a diferença em cada seed e então a média das três diferenças pareadas. A incerteza é condicional aos checkpoints congelados e não incorpora integralmente a variação de treinamento.

## Testes pareados e multiplicidade

Usar 20.000 permutações de troca no nível da unidade: trocar conjuntamente as previsões dos dois modelos para todos os objetos da unidade e todos os três seeds. O p-valor bilateral de randomização é `(1 + # permutações tão ou mais extremas)/(20.001)`. A mesma sequência de trocas é aplicada às métricas em cada comparação.

Holm step-down, alfa 0,05, em famílias separadas: (i) três pares de arquiteturas × três métricas na representação primária A (9 hipóteses); (ii) os mesmos pares × três métricas em B, A_outer e B_outer (27 hipóteses); (iii) cada controle B/A_outer/B_outer contra A × três arquiteturas × três métricas (27 hipóteses). Intervalos percentis são marginais, não simultâneos após Holm. Não usar teste t com n=3 seeds.

Os dois modelos metadata/FOV fixos são sempre relatados com as mesmas métricas e intervalos pareados por unidade contra cada arquitetura em A. Esses resultados são controles descritivos; não participam da seleção e não podem promover outra representação.

## Componentes com múltiplos objetos

Preservar previsões independentes por objeto e todas as probabilidades. Não colapsar em majority vote, não subdividir unidades e não usar 73 como número de réplicas independentes. Bootstrap/teste sempre sorteiam unidades; métricas por estrato FOV/extensão reportam suporte e resultados indefinidos, inclusive bins vazios.
'''
(R/'statistical_analysis_plan.md').write_text(stats)
opens='''# Decisões autorais pendentes antes da execução

1. **Escolher A ou B para seleção de configuração.**
   - **A:** importar sem retuning os três vencedores Phase 4C. São 36 fits de redes (12 por arquitetura) mais dois fits fixos do controle metadata/FOV: **38 fits no total**.
   - **B:** busca apenas em A/validation sobre os grids arquivados (12 configs ResNet50, 12 ViT-B/16 e seis configs custom que contêm Transformer). Selecionar um config por arquitetura e aplicar esse config sem alteração a B, A_outer e B_outer. São 117 fits de redes (90 fits de busca em A + 27 controles condicionais) mais dois fits metadata/FOV: **119 fits no total**. As 360 vagas possíveis de rede estão enumeradas.
   Os protocolos v1 não determinam se o conjunto objeto-disjunto deve herdar os vencedores v1 ou receber uma nova busca prospectiva.

2. **Resolver a entrada do Custom CNN+Transformer.** O vencedor v1 usa `input_size=256`; v2H congela tensores 224×224, e o modelo v1 forma uma grade 8×8 após o pooling. Alternativas: (a) manter a arquitetura e congelar uma adaptação determinística explícita 224→256; ou (b) congelar uma variante para 224 (por exemplo, grade 7 e regra posicional/pooling compatível), aceitando que não é a arquitetura efetiva v1. Nenhum protocolo existente decide isso.

3. **Aceitar a faixa de runtime.** ResNet e ViT têm tempos v1 medidos na RTX 5060 Ti. Os registros de treino custom não preservam tempos por época; a estimativa custom é uma proxy ampla de ViT, não uma medição. Se essa incerteza não servir para orçamento, uma estimativa de engenharia deverá ser aprovada antes da campanha. Não executar um fit-piloto nesta fase.

4. **Confirmar a transferência ImageNet para blue/red/NIR.** A proposta mantém os pesos oficiais v1 com hashes fixos, injeta blue/red/NIR diretamente nos três canais e usa os tensores H sem normalização adicional específica da família. Isso preserva a representação congelada, embora os canais não tenham semântica RGB convencional. A alternativa seria treinar baselines do zero, mudando a política de inicialização. É necessária decisão explícita.

5. **Confirmar o executor estatístico.** O plano usa as 61 unidades de atribuição v2G, bootstrap PCG64 seed 20261021 e 20.000 replicatas. O executor futuro deve receber as identidades das unidades depois que previsões estiverem seladas e antes do scoring, sem disponibilizar truth ao worker.

`ready_for_execution=false` até decisões explícitas em 1, 2 e 4 e confirmação operacional em 3 e 5. Nenhuma decisão pode usar métricas de teste.
'''
(R/'open_decisions.md').write_text(opens)

candidate_sha=sha(R/'candidate_manifest.json'); accounting_sha=sha(R/'fit_accounting.json')
protocol={
 'phase':'Dataset-v2I','status':'prospective_protocol_frozen_pending_author_choices','ready_for_execution':False,
 'source_digests':{
  'v2F_representation':sha(F/'representation_protocol.json'),'v2F_shortcut_controls':sha(F/'shortcut_control_protocol.json'),
  'v2G_split':sha(G/'split_manifest.json'),'v2H_inventory':sha(H/'file_inventory.json'),
  'v2H_dataset_digest':json.loads((H/'dataset_digest.json').read_text())['canonical_dataset_sha256'],
  'v2H_tensor_merkle':json.loads((H/'dataset_digest.json').read_text())['tensor_merkle_root'],
  'v1_protocol':sha(V1/'protocol.json'),'v1_phase4b_selection':sha(P4B/'selection.json'),
  'v1_phase4c_selection':sha(P4C/'frozen_selection.json'),'v1_phase4c_baseline_audit':sha(P4C/'baseline_audit.json'),
  'v1_phase4c_custom_audit':sha(P4C/'custom_audit.json'),'v1_phase4e_statistical_protocol':sha(P4E/'analysis_protocol.json')},
 'candidate_manifest_sha256':candidate_sha,'fit_accounting_sha256':accounting_sha,
 'statistical_analysis_plan_sha256':sha(R/'statistical_analysis_plan.md'),'open_decisions_sha256':sha(R/'open_decisions.md'),
 'scope':{'task':'four-class astronomical classification','classes':['galaxies','globularclusters','nebulae','opencluster'],
  'architectures':['custom_cnn_transformer','resnet50','vit_b_16'],'primary_representation':'A',
  'mandatory_controls':['B','A_outer','B_outer','metadata_FOV_only','FOV_stratified_reporting'],
  'taxonomy_aware_or_hierarchical_classification':False,'statistical_unit':'v2G assignment unit / overlap component'},
 'data_and_preprocessing':{
  'split_sha256':'1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf',
  'role_counts_objects':{'train':222,'validation':75,'test':73},'test_assignment_units':61,
  'source':'v2H tensors only; one object-level example with blue/red/NIR channel order',
  'normalization':'Use v2H per-object p01/p99, clipping, asinh(10), and fixed (u-0.5)/0.5 exactly. No additional fitted image normalization. Any metadata imputation/scaling uses training objects only.',
  'class_weights':'Reciprocal training-object counts by class, float32, normalized by float32 mean; computed from train labels only and shared across all families/representations.',
  'class_order':['galaxies','globularclusters','nebulae','opencluster']},
 'augmentation':{
  'recipe':'v2F deterministic D4 keyed by SHA256 UTF-8 Dataset-v2F|augment|2026102303|physical_object_id|zero_based_epoch; first 8 bytes big endian modulo 8. Codes 0–3 are quarter-turn CCW; 4–7 mirror horizontally then rotate.',
  'train_only':True,'same_object_epoch_transform_for_all_channels_representations_architectures_and_seeds':True,
  'validation_test':'identity','prohibited':['random crop','continuous rotation','brightness/contrast/color jitter','cutout','noise','mixup','test-time augmentation'],
  'implementation_gate':'Before training, prove an implementation over stored v2H tensors is numerically equivalent to the frozen source-lattice operation; otherwise apply the transform on the frozen A 512 and B 16 lattices before sizing. Do not pre-materialize random views.'},
 'determinism':{'fit_seeds':[3908274,3908275,3908276],'seed_Python_NumPy_Torch_CPU_and_all_CUDA_generators_per_fit':True,
  'DataLoader':'seeded generator, deterministic worker initialization, fixed shuffle algorithm; validation and test order do not shuffle',
  'cuda':'CUBLAS_WORKSPACE_CONFIG=:4096:8; deterministic algorithms and cuDNN deterministic; benchmark false; TF32 disabled',
  'seed_unit':'same numeric seed matches corresponding architecture/configuration/representation runs; seeds are training replicates, not an inferential n=3 sample'},
 'initialization':{
  'custom_cnn_transformer':'Seeded source-defined initialization; no checkpoint reuse.',
  'resnet50':{'policy':'official torchvision ResNet50_Weights.IMAGENET1K_V1, exact local SHA-256, no network; missing or mismatched file stops execution.','sha256':'0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a'},
  'vit_b_16':{'policy':'official torchvision ViT_B_16_Weights.IMAGENET1K_V1, exact local SHA-256, no network; missing or mismatched file stops execution.','sha256':'c867db91d3e12c6cbadabb610d73c24a546bf82d8c03a9fea34f43a712ddb0e9'},
  'channel_policy':'Pass blue/red/NIR directly into the three channel slots; do not permute or synthesize channels. Use H values without additional ImageNet or family-specific normalization.'},
 'optimization_proposal':{
  'seeds':[3908274,3908275,3908276],'max_epochs':100,'batch_size':16,'workers':8,'cpu_threads':2,'gradient_clip_norm':3.0,
  'weighted_loss':'class-weighted cross-entropy; mean reduction over objects',
  'scheduler':{'type':'ReduceLROnPlateau','monitor':'weighted validation loss','factor':0.1,'patience':5,'active_from_epoch':1,'no_reset_at_unfreeze':True},
  'early_stopping':{'monitor':'weighted validation loss','patience':20,'min_delta':0.0},
  'family_settings':{
   'custom_cnn_transformer':{'optimizer':'Adam','lr':9e-5,'weight_decay':5e-7,'betas':[0.9,0.999],'epsilon':1e-8},
   'resnet50':{'optimizer':'AdamW','backbone_lr':1e-4,'head_lr_multiplier':10,'weight_decay':1e-4,'betas':[0.9,0.999],'epsilon':1e-8,'head_only_warmup_epochs':5,'one_optimizer_no_reset':True,'backbone_batchnorm_eval_while_frozen':True},
   'vit_b_16':{'optimizer':'AdamW','backbone_lr':3e-5,'head_lr_multiplier':10,'weight_decay':1e-2,'betas':[0.9,0.999],'epsilon':1e-8,'head_only_warmup_epochs':5,'one_optimizer_no_reset':True}},
  'checkpoint_and_resume':'Save epoch-boundary last state including model, optimizer, scheduler, RNG and sampler states; preserve best state at minimum sample-mean weighted validation loss. Resume only from hash-verified complete state with identical configuration, protocol, source and environment. Never restart due to poor performance or reset optimizer/scheduler.',
  'failure':'OOM, nondeterminism, missing pretrained weights, corrupt state or nonfinite loss stops the fit/campaign. No batch, precision, seed, preprocessing, architecture or optimizer changes to rescue a run; no replacement fit without new authorization.',
  'resource_limits':{'one_exclusive_GPU':'RTX 5060 Ti class as in v1 execution; no concurrent fits','precision':'float32 only; CUDA deterministic settings inherited from v1','memory':'record peak allocated/reserved; stop at 90% of device memory; no batch fallback','software_target':'Python 3.12.3; torch 2.7.0+cu128; torchvision 0.22.0+cu128; freeze full environment/source digest before launch'}},
 'model_selection':{
  'validation_only':True,'checkpoint_within_fit':'minimum sample-mean class-weighted validation loss, all three seeds',
  'alternative_A':'Use the exact Phase 4C winner per architecture without v2 hyperparameter search.',
  'alternative_B':'Search v2 validation using archived candidate grids on A only. Rank by mean balanced accuracy, then mean macro-F1, lowest mean weighted validation loss, lowest sample variance (ddof=1) of balanced accuracy, then lexicographically smallest candidate ID. Select one configuration per architecture and transfer it unchanged to all controls.',
  'best_seed_selection':False,'test_access_before_selection_seal':False,'test_based_changes':False},
 'factorial':{'design':'3 architectures × 4 representations × 3 seeds','selected_fit_count':36,'selected_test_evaluation_count':36,
  'identical_architecture_configuration_across_representations':True,'primary':'A','mandatory_controls':['B','A_outer','B_outer','metadata_FOV_only','FOV_stratified_reporting'],
  'retrospective_representation_selection':False},
 'metadata_FOV_control':{'fit_count':2,'models':[{'type':'multinomial_logistic_regression','C':1,'class_weight':'balanced','max_iter':1000,'solver':'lbfgs'},
  {'type':'decision_tree','max_depth':3,'random_state':3908274}],'features':['log(source_FOV_arcmin)','log(source_pixel_scale_arcsec)','log(catalogue_extent_arcmin)','catalogue_extent_missing','source_FOV_capped'],
  'imputation_and_scaling':'training objects only; median imputation and centering/scaling; no held-out fitting','purpose':'mandatory leakage/shortcut control; never selects representations or model configs'},
 'statistical_analysis':{'metrics':['balanced_accuracy','macro_f1','accuracy','per_class_precision_recall_f1_support','confusion_matrix'],
  'metric_conventions':'balanced accuracy averages recall over classes with positive true support; macro-F1 includes all four frozen classes with zero when denominator is zero; report degeneracies/support explicitly',
  'retain_each_object_prediction':True,'assignment_unit_count_test':61,'bootstrap_replicates':20000,'bootstrap_seed':20261021,
  'bootstrap_unit':'resample 61 assignment units with replacement; keep all objects in each unit; same draw for every paired model, representation and seed',
  'seed_summary':'report seed-specific metrics and paired differences; report every seed and mean ± sample SD (ddof=1); never best seed',
  'paired_permutation_replicates':20000,'multiplicity':'Holm alpha .05: 9 primary A architecture-pair×metric tests; separate 27-test family for architecture contrasts on controls and separate 27-test family for each control versus A.',
  'metadata_control':'Report both fixed metadata models and paired cluster-bootstrap differences versus each primary A architecture descriptively; no representation selection or tuning from these controls.'},
 'test_lock_policy':{'worker_has_truth':False,'test_evaluations':38,'deep_checkpoint_evaluations':36,'metadata_control_evaluations':2,'observations_per_checkpoint':73,'seal_predictions_before_scoring':True,
  'one_final_pass_only':True,'test_results_cannot_trigger_retraining_replacement_or_reselection':True},
 'decisions_pending':['selection alternative A or B','custom input-size compatibility','accept transfer-time uncertainty and budget','ImageNet transfer policy for blue/red/NIR','future test-lock executor validation'],
 'execution_prohibited_now':True}
protocol['canonical_sha256']=hashlib.sha256(canon(protocol)).hexdigest()
dump('protocol.json',protocol)

am=json.loads((H/'representation_A_manifest.json').read_text())
test_rows=[x for x in am['records'] if x['split']=='test']
assert len(test_rows)==73 and len({x['split_unit_id'] for x in test_rows})==61
test_membership_digest=hashlib.sha256(canon([{'physical_object_id':x['physical_object_id'],'split_unit_id':x['split_unit_id'],'path':x['path'],'sha256':x['sha256']} for x in test_rows])).hexdigest()
test_lock={'phase':'Dataset-v2I','status':'policy_locked_execution_not_authorized','test_access_allowed_now':False,
 'test_label_manifest_read':False,'test_membership_identifiers_read_for_integrity':True,
 'test_label_manifest_sha256':next(x['sha256'] for x in json.loads((H/'file_inventory.json').read_text())['files'] if x['path']=='sealed_label_manifest.json'),
 'test_split_sha256':protocol['data_and_preprocessing']['split_sha256'],'test_membership_digest':test_membership_digest,'test_observation_count':73,'test_assignment_unit_count':61,
 'training_protocol_sha256':protocol['canonical_sha256'],'candidate_manifest_sha256':candidate_sha,
 'final_evaluations_if_later_authorized':38,'deep_checkpoint_evaluations':36,'metadata_control_evaluations':2,'one_evaluation_per_selected_checkpoint':True,
 'seal_order':['freeze ordered test membership identity without truth','run all 36 selected checkpoint evaluations and two metadata-control evaluations once','seal/hash every prediction record','only then release truth to independent scoring','compute metrics and paired analysis once'],
 'never_reselect_or_replace_after_test':True}
dump('test_lock.json',test_lock)
validation={'phase':'Dataset-v2I','status':'passed_planning_with_author_decisions_pending','ready_for_execution':False,
 'v2H_files_hash_verified':1496,'v2A_v2G_files_preserved_and_verified':9209,'sealed_test_labels_deserialized':False,
 'test_predictions_or_Phase4D_outcomes_read':False,'v2H_tensor_arrays_loaded_or_reconstructed':False,
 'candidate_manifest_has_results_or_prediction_hashes':False,'A_core_fit_count':36,'A_total_fit_count':38,'B_core_expected_fit_count':117,'B_total_expected_fit_count':119,
 'B_potential_core_slots':360,'metadata_control_fits':2,'selected_factorial_test_evaluations':36,'one_time_total_test_evaluations':38,'author_decisions_pending':5,
 'training_inference_checkpoint_loading_or_scoring_performed':False}
dump('validation.json',validation)
report='''# Dataset-v2I — protocolo prospectivo de treino e análise

## Estado

**Não pronto para execução (`ready_for_execution=false`).** O plano e a enumeração são determinísticos, mas decisões autorais sobre seleção, compatibilidade dimensional do custom e transferência ImageNet ainda são necessárias. Nenhum treino, inferência, carregamento de checkpoint, acesso aos rótulos de teste ou scoring ocorreu.

## Escopo e integridade

O benchmark é de classificação de quatro classes (galáxias, globulares, nebulosas e aglomerados abertos), com Custom CNN+Transformer, ResNet50 e ViT-B/16. A é primária; B, A_outer, B_outer, metadata/FOV-only e estratificação por FOV são controles obrigatórios em qualquer resultado. Classificação hierárquica está fora do escopo.

O split v2G vinculante é `1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf`. A verificação por streaming confirmou os 1.496 arquivos do inventário v2H e os 9.209 arquivos preservados de v2A–v2G. O hash do manifesto de rótulos foi conferido sem desserializar seu conteúdo. Arrays dos tensores não foram carregados nem reconstruídos.

## Duas opções de seleção

- **A — reutilizar winners v1 sem retuning:** 36 fits de redes (3 arquiteturas × 4 representações × 3 seeds) mais dois fits fixos metadata/FOV — **38 fits totais**. Estimativa central de wall time GPU para redes **5,3 h**, faixa indicativa **3,3–14,8 h**.
- **B — busca v2 em validação:** 90 fits de busca em A (12 candidatos ResNet × 3 seeds, 12 ViT × 3 seeds, seis custom Transformer × 3 seeds), mais 27 fits de controle para os três winners e dois metadata/FOV — **119 fits totais**. Estimativa central para redes **16,5 h**, faixa indicativa **10,9–40,8 h**. O manifesto lista 360 slots possíveis de rede; somente 90 de busca e os nove slots de controle do vencedor por arquitetura seriam executados. Os dois ajustes metadata são fixos e CPU; custo não incluído na estimativa GPU.

As estimativas usam tempos por época v1 na RTX 5060 Ti. Baselines são escalados por 222/485 observações de treino e 75/158 de validação para o cenário central; a faixa alta usa o percentil 90 sem redução por tamanho. O v1 não preservou tempos de época do modelo custom: sua faixa usa proxy ViT com fator 0,5×–2× e limite superior de 100 épocas. Os intervalos são planejamento, não garantia nem medição v2. Não incluem inicialização, escrita de checkpoints e inferência final.

Os grids vêm dos registros v1 congelados. A opção B seleciona somente na validação A pela regra proposta em `protocol.json`, com os três seeds; config vencedor é mantido nos controles. Checkpoints por fit são escolhidos pelo menor loss ponderado de validação. Nunca se escolhe o melhor seed.

## Treino e análise congelados como proposta

O plano retém tensores v2H e normalização fixa, D4 determinístico só no treino, pesos oficiais ImageNet com hashes v1 para ResNet/ViT e inicialização seeded para custom. O seed de cada fit controla Python, NumPy, Torch CPU/CUDA, sampler e workers; a transformação D4 usa o fluxo de objeto/época v2F. Batch 16, no máximo 100 épocas, early stopping de 20 épocas, scheduler ReduceLROnPlateau, float32, CUDA determinístico e um GPU exclusivo. Falhas não permitem alterar batch, parâmetros ou representação; somente resume exato de estado completo é aceito.

O desenho selecionado terá **36 fits/checkpoints de redes** (3×4×3), mais dois fits fixos metadata/FOV. A avaliação final única terá 36 passagens de checkpoint e duas passagens metadata (38 no total); cada saída cobre todos os 73 objetos, com 2.628 previsões objeto-checkpoint das redes. Previsões individuais são seladas antes de liberar truth ao scoring. Métricas primárias: balanced accuracy, macro-F1 e accuracy; também precisão/recall/F1 por classe e matrizes de confusão.

Incerteza e testes pareados usam as 61 unidades de atribuição v2G, não 73 objetos independentes. Cada bootstrap de 20.000 replicatas amostra unidades completas, retém todos os seus objetos e reutiliza a mesma amostra entre os modelos. A análise não colapsa as previsões de objetos do mesmo grupo; evita pseudorreplicação na inferência por reamostragem de clusters. Plano estatístico e correção Holm constam em `statistical_analysis_plan.md`.

## Decisões em aberto

1. Escolher A (36 fits de rede) ou B (117 fits de rede; busca v2 em A/validation), além dos dois ajustes fixos metadata/FOV.
2. Definir como a entrada v1 custom de 256 pixels será compatibilizada com o tensor v2H de 224 sem alterar a representação A.
3. Aceitar o intervalo de runtime custom proxy (não medido) ou exigir outra estimativa de engenharia antes do orçamento.
4. Confirmar o uso de pesos ImageNet em blue/red/NIR, mantendo H sem normalização adicional, ou optar por inicialização do zero — que alteraria a política v1.
5. Validar o executor do test lock, incluindo identidade de unidades sem entregar truth ao worker.

As alternativas e consequências estão em `open_decisions.md`. A execução permanece bloqueada até resolução explícita.

O arquivo `candidate_manifest.json` enumera fits prospectivos por IDs e configuração; não contém resultados, hashes de previsões nem tempos fabricados. `test_lock.json` proíbe acesso atual a teste e fixa a ordem futura de selagem. Resultados da Fase 4D não foram lidos.
'''
(R/'REPORT.md').write_text(report)
print('protocol outputs written; ready_for_execution=false')
