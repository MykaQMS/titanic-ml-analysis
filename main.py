import os
import sys
import pandas as pd

# Suporte para caracteres no Windows Console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.processamento import preparar_dados
from src.modelagem import treinar_e_comparar_modelos, gerar_graficos_eda, gerar_grafico_importancia

def main():
    print("="*65)
    print("🚢 PROJETO MACHINE LEARNING: PREVENDO SOBREVIVENTES DO TITANIC")
    print("="*65)
    
    # 1. Carregando dados brutos
    caminho_treino = os.path.join("data", "train.csv")
    caminho_teste = os.path.join("data", "test.csv")
    
    if not os.path.exists(caminho_treino) or not os.path.exists(caminho_teste):
        print("❌ Erro: Arquivos train.csv ou test.csv não encontrados na pasta 'data/'")
        return
        
    df_treino = pd.read_csv(caminho_treino)
    df_teste = pd.read_csv(caminho_teste)
    print(f"\n[1/5] Dados carregados com sucesso! (Treino: {df_treino.shape[0]} passageiros, Teste: {df_teste.shape[0]} passageiros)")
    
    # 2. Gerar Gráficos de Análise Exploratória (EDA)
    print("\n[2/5] Gerando gráficos da Análise Exploratória (EDA)...")
    gerar_graficos_eda(df_treino, pasta_figuras="reports/figuras")
    print("  └─ Gráficos salvos na pasta 'reports/figuras/'")
    
    # 3. Pré-processamento e Engenharia de Atributos
    print("\n[3/5] Aplicando tratamento de dados e Engenharia de Features...")
    X_treino, y_treino, X_teste, ids_teste = preparar_dados(df_treino, df_teste)
    print(f"  └─ Dados prontos para o modelo! ({X_treino.shape[1]} atributos numéricos/dummies criados)")
    
    # 4. Treinamento e Comparação de Modelos
    print("\n[4/5] Treinando modelos e comparando a acurácia via Validação Cruzada (5 Folds)...")
    df_resultados, modelos = treinar_e_comparar_modelos(X_treino, y_treino)
    
    melhor_nome = df_resultados.iloc[0]['Modelo']
    melhor_acuracia = df_resultados.iloc[0]['Acurácia Média']
    print(f"\n🏆 Modelo com melhor resultado: {melhor_nome} (Acurácia: {melhor_acuracia*100:.2f}%)")
    
    # 5. Treinamento Final e Geração da Submissão Kaggle
    print("\n[5/5] Treinando o melhor modelo com todos os dados e gerando submissão...")
    modelo_final = modelos[melhor_nome]
    modelo_final.fit(X_treino, y_treino)
    
    # Se o melhor modelo tiver importância de atributos, gera o gráfico
    if hasattr(modelo_final, 'feature_importances_'):
        gerar_grafico_importancia(modelo_final, X_treino.columns, pasta_figuras="reports/figuras")
        print("  └─ Gráfico de importância de atributos salvo em 'reports/figuras/importancia_atributos.png'")
        
    predicoes = modelo_final.predict(X_teste)
    
    os.makedirs("submissions", exist_ok=True)
    df_submissao = pd.DataFrame({
        'PassengerId': ids_teste,
        'Survived': predicoes
    })
    
    caminho_submissao = os.path.join("submissions", "submission.csv")
    df_submissao.to_csv(caminho_submissao, index=False)
    
    print(f"\n✅ Arquivo de submissão criado com sucesso em: {caminho_submissao}")
    print("="*65)
    print("🎉 PROCESSO CONCLUÍDO COM SUCESSO!")
    print("="*65)

if __name__ == "__main__":
    main()
