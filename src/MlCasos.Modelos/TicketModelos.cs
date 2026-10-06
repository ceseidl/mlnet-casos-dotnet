using Microsoft.ML.Data;

namespace MlCasos.Modelos;

// Entrada: uma linha do tickets.tsv
// (Categoria só é preenchida no treino).
public class TicketEntrada
{
    [LoadColumn(0)] public string Categoria { get; set; } = "";
    [LoadColumn(1)] public string Texto { get; set; } = "";
}

// Saída: a categoria prevista e a probabilidade de cada classe.
public class TicketSaida
{
    [ColumnName("PredictedLabel")]
    public string Categoria { get; set; } = "";

    public float[] Score { get; set; } = [];
}
