# Telecom X — Análise de Evasão de Clientes (Churn)

Resumo executivo, gráficos recomendados e instruções para executar e reproduzir a análise do notebook de modelagem.

---

## Sumário

- Objetivo
- Dataset e preparação
- Exploração (EDA) e gráficos
- Modelos e avaliação
- Importância das variáveis (por modelo)
- Principais fatores de churn
- Estratégias de retenção
- Como executar (Colab e local)
- Estrutura do projeto
- Reprodutibilidade
- Próximos passos

---

## Objetivo

Prever evasão de clientes (churn) e identificar os fatores que mais influenciam o cancelamento, orientando ações práticas de retenção.

---

## Dataset e preparação

- Segmento: Telecom (clientes, planos, cobrança, serviços adicionais).
- Alvo: coluna binária de cancelamento (0 = ativo, 1 = cancelou).
- Pré-processamento:
  - Limpeza de nulos e consistência de tipos.
  - One-hot encoding para categorias (ex.: Tipo de Contrato, Método de Pagamento).
  - Padronização para modelos baseados em distância/hiperplanos (KNN, SVM, Regressão Logística).
  - Split estratificado em treino/teste.

Nota técnica: reduzir multicolinearidade removendo uma entre Cobrança Mensal e variáveis derivadas fortemente correlacionadas.

---

## Exploração (EDA) e gráficos

Gráficos recomendados (paths sugeridos):
- Distribuição do alvo (churn_rate.png)
- Relação churn × tempo de permanência (tenure_vs_churn.png)
- Churn por tipo de contrato (contract_vs_churn.png)
- Boxplots de cobrança mensal por status (monthlycharges_box.png)
- Heatmap de correlação (correlation_heatmap.png)
- Importância de variáveis por modelo (ex.: featimp_rf.png, coefs_lr.png)

Exemplo de geração e salvamento (Matplotlib/Seaborn):

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Churn rate
ax = y.value_counts(normalize=True).mul(100).rename({0:'Ativos',1:'Churn'}).plot(kind='bar', color=['#3b82f6','#ef4444'])
plt.title('Distribuição do Alvo (%)'); plt.ylabel('%'); plt.tight_layout()
plt.savefig('images/churn_rate.png', dpi=120)

# Boxplot MonthlyCharges por churn
sns.boxplot(x=y, y=X['MonthlyCharges'])
plt.title('Cobrança Mensal por Status'); plt.xlabel('Churn (0=Não, 1=Sim)'); plt.tight_layout()
plt.savefig('images/monthlycharges_box.png', dpi=120)
```

---

## Modelos e avaliação

Foram treinados e comparados:

- Regressão Logística (com normalização)
- Random Forest (sem normalização)

Resultados no conjunto de teste:
- Regressão Logística: acurácia 0.801, precisão 0.6348, recall 0.533, F1 0.5795
- Random Forest: acurácia 0.7992, precisão 0.6783, recall 0.4171, F1 0.5166

Matrizes de confusão:
- Regressão Logística:
  - [[1448, 172], [262, 299]]
- Random Forest:
  - [[1509, 111], [327, 234]]

Conclusão de escolha: Regressão Logística preferida para retenção (maior F1 e recall, capturando mais clientes em risco).

---

## Importância das variáveis (por modelo)

### Regressão Logística (coeficientes; padronizada)
- Tipo de Contrato: Mês a mês — aumenta risco (coeficiente positivo)
- Tempo de Permanência — reduz risco (coeficiente negativo)
- Cobrança Mensal — aumenta risco (positivo)
- Suporte Técnico: Não — aumenta risco
- Método de Pagamento: Boleto impresso — aumenta risco
- Engajamento (Streaming TV/Filmes) — tende a reduzir risco

Visual: gráfico de barras dos top coeficientes (coefs_lr.png).

### Random Forest (redução de impureza + permutação)
- Tempo de Permanência — alta importância
- Tipo de Contrato: Mês a mês — alta importância
- Cobrança Mensal — alta importância
- Suporte Técnico — relevante
- Método de Pagamento: Boleto impresso — relevante

Visual: feature_importances_ ordenadas (featimp_rf.png) e validação por permutação.

### KNN (permutação; sensível à normalização)
- Cobrança Mensal e Tempo de Permanência dominam por influenciarem distâncias
- Sinais de engajamento (Streaming) frequentemente aparecem como protetores

### SVM
- Linear: coeficientes destacam Tipo de Contrato (↑), Cobrança Mensal (↑), Tempo de Permanência (↓)
- Kernel RBF: use permutação/SHAP; mesmo conjunto de variáveis tende a emergir

### Boosting/Outros (ex.: XGBoost, MLP)
- XGBoost (gain): Tempo de Permanência, Tipo de Contrato, Cobrança Mensal, Pagamento (Boleto) no topo
- MLP: interpretar via permutação/SHAP; padrões semelhantes aos demais

---

## Principais fatores de churn (consenso entre modelos)

- Contrato Mês a mês — maior propensão à evasão
- Permanência baixa — risco elevado nos primeiros meses
- Cobrança Mensal alta — sensibilidade a preço/valor percebido
- Ausência de Suporte Técnico — indica fricção e maior cancelamento
- Boleto impresso — perfis com maior probabilidade de churn
- Engajamento em serviços (Streaming, etc.) — fator protetivo

---

## Estratégias de retenção

- Fidelização e contratos mais longos:
  - Descontos progressivos e bônus de serviços para migração de mês a mês → anual/bienal
- Onboarding e suporte proativo:
  - Suporte dedicado nos primeiros 90 dias e check-ins programados
- Política de preços e planos flexíveis:
  - “Pague pelo que usar”, revisão de cobranças elevadas, upgrades/downgrades sem fricção
- Pagamentos e conveniência:
  - Incentivar débito automático/app; reduzir dependência de boleto impresso
- Engajamento em serviços adicionais:
  - Trials e bundles de Streaming/Backup/Segurança digital
- Operação orientada a risco:
  - Ajustar limiar de decisão do modelo para priorizar recall
  - Ações proativas para top-N clientes de maior propensão

---

## Como executar

### No Google Colab
1. Abra o notebook `clientes_telecom_parte_2.ipynb`.
2. Execute as células em ordem (Runtime → Run all).
3. Ao final, salve gráficos em `images/` e exporte resultados (predições/relatórios) conforme células do notebook.

### Local (Python 3.10+)

Requisitos mínimos (requirements.txt sugerido):
```
pandas
numpy
scikit-learn
matplotlib
seaborn
imblearn
xgboost
shap
```

Instalação e execução:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Execute scripts/notebooks
jupyter notebook clientes_telecom_parte_2.ipynb
```

---

## Estrutura do projeto

```
.
├─ clientes_telecom_parte_2.ipynb
├─ images/
│  ├─ churn_rate.png
│  ├─ tenure_vs_churn.png
│  ├─ contract_vs_churn.png
│  ├─ monthlycharges_box.png
│  ├─ correlation_heatmap.png
│  ├─ coefs_lr.png
│  └─ featimp_rf.png
├─ data/                # (opcional) dados brutos e tratados
├─ models/              # (opcional) artefatos treinados (.pkl)
├─ requirements.txt
└─ README.md
```

---

## Reprodutibilidade

- Split estratificado e `random_state` fixo para modelos.
- Relatar versões das bibliotecas (pip freeze > requirements.txt).
- Salvar métricas e gráficos versionados no repositório.
- Descrever qualquer tratamento de desbalanceamento (ex.: class_weight, SMOTE) e thresholds usados.

---

## Próximos passos

- Calibração de limiar para maximizar F1/Recall conforme custo de falso negativo.
- Teste de boosting (XGBoost/LightGBM) e ensemble simples (votação/stacking).
- Monitoramento de drift e re-treino trimestral.
- Dashboards de acompanhamento de churn por segmento e coorte.
