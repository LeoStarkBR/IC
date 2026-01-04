import pandas as pd
import numpy as np

# Gera 150 linhas de dados sujos para teste
data = {
    'id_caso': range(1, 151),
    'data_notificacao': pd.date_range(start='2024-01-01', periods=150).strftime('%Y-%m-%d'),
    'bairro': np.random.choice(['Centro', 'Vila Falcão', 'Mary Dota', 'Geisel', 'Redentor'], 150),
    'status_caso': np.random.choice(['Confirmado', 'Suspeito', 'Descartado'], 150)
}
df = pd.DataFrame(data)
# Salva o arquivo que vamos usar para upload
df.to_csv('dados_dengue_bauru_sujos.csv', index=False)
print("Arquivo criado com sucesso!")