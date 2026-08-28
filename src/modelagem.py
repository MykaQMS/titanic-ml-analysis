import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Estilo visual limpo para os gráficos
sns.set_theme(style='whitegrid')

def treinar_e_comparar_modelos(X_treino: pd.DataFrame, y_treino: pd.Series):
    """
    Treina múltiplos algoritmos de classificação e compara suas acurácias
    usando Validação Cruzada (Cross-Validation de 5 Folds).
    """
    modelos = {
        'Regressão Logística': LogisticRegression(max_iter=1000, random_state=42),
        'Árvore de Decisão': DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42, eval_metric='logloss')
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    resultados = []
    
    print("\n" + "="*60)
    print(" 🤖 AVALIAÇÃO E COMPARAÇÃO DE MODELOS (CROSS-VALIDATION)")
    print("="*60)
    
    for nome, modelo in modelos.items():
        scores = cross_val_score(modelo, X_treino, y_treino, cv=cv, scoring='accuracy')
        acuracia_media = scores.mean()
        desvio_padrao = scores.std()
        
        resultados.append({
            'Modelo': nome,
            'Acurácia Média': acuracia_media,
            'Desvio Padrão': desvio_padrao
        })
        
        print(f"-> {nome:20s}: Acurácia = {acuracia_media*100:.2f}% (± {desvio_padrao*100:.2f}%)")
        
    df_resultados = pd.DataFrame(resultados).sort_values(by='Acurácia Média', ascending=False).reset_index(drop=True)
    return df_resultados, modelos

def gerar_graficos_eda(df_treino: pd.DataFrame, pasta_figuras: str = "reports/figuras"):
    """
    Gera e salva gráficos fundamentais para entender o comportamento dos dados.
    """
    os.makedirs(pasta_figuras, exist_ok=True)
    df = df_treino.copy()
    
    # 1. Sobrevivência por Sexo
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='Sex', y='Survived', hue='Sex', palette='Set2', legend=False, ax=ax)
    ax.set_title('Taxa de Sobrevivência por Sexo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proporção de Sobreviventes (0 a 1)')
    ax.set_xlabel('Sexo')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Masculino', 'Feminino'])
    plt.tight_layout()
    fig.savefig(os.path.join(pasta_figuras, 'sobrevivencia_por_sexo.png'), dpi=300)
    plt.close(fig)
    
    # 2. Sobrevivência por Classe
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df, x='Pclass', y='Survived', hue='Pclass', palette='crest', legend=False, ax=ax)
    ax.set_title('Taxa de Sobrevivência por Classe do Bilhete', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proporção de Sobreviventes (0 a 1)')
    ax.set_xlabel('Classe (1ª, 2ª, 3ª)')
    plt.tight_layout()
    fig.savefig(os.path.join(pasta_figuras, 'sobrevivencia_por_classe.png'), dpi=300)
    plt.close(fig)
    
    # 3. Distribuição de Idade por Sobrevivência
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.kdeplot(data=df, x='Age', hue='Survived', common_norm=False, fill=True, palette=['#e74c3c', '#2ecc71'], alpha=0.5, ax=ax)
    ax.set_title('Distribuição de Idade: Sobreviventes vs Não-Sobreviventes', fontsize=12, fontweight='bold')
    ax.set_xlabel('Idade (Anos)')
    ax.set_ylabel('Densidade')
    plt.tight_layout()
    fig.savefig(os.path.join(pasta_figuras, 'distribuicao_idade.png'), dpi=300)
    plt.close(fig)

def gerar_grafico_importancia(modelo, colunas, pasta_figuras: str = "reports/figuras"):
    """
    Gera o gráfico mostrando quais variáveis foram mais importantes para a previsão.
    """
    os.makedirs(pasta_figuras, exist_ok=True)
    
    importancias = pd.Series(modelo.feature_importances_, index=colunas).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    importancias.tail(10).plot(kind='barh', color='#2b5c8f', edgecolor='black', ax=ax)
    ax.set_title('Top 10 Atributos Mais Importantes (Random Forest)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Relevância Relativa')
    ax.set_ylabel('Atributo')
    plt.tight_layout()
    fig.savefig(os.path.join(pasta_figuras, 'importancia_atributos.png'), dpi=300)
    plt.close(fig)
