import importlib
import metaheuristic
import pandas as pd

importlib.reload(metaheuristic)

reproduccion = [
    "699 348.3064 0.9294 0.5788 0.1834 0.0113 0.8166 0.6073 0.5 0.7267 0.0362 0.9632 0.4424 0.8195 0.3605 0.573 0.218 0.3627 0.0611 1 0.6984 0.567 0.3064 0.2407 0.152 0.4833 8 0.2649 0.7624 0.5197",
    "3111 444.7435 0.9268 0.8329 0.2286 0.3525 0.7626 0.5038 0.4984 0.4353 0.0254 0.856 0.347 0.7623 0.3378 0.6336 0.288 0.4949 0.0746 1 0.7887 0.3848 0.4197 0.2041 0.2058 0.3865 7 0.0136 0.9698 0.3936",
    "5408 1066.0777 0.9037 0.7473 0.1972 0.5913 0.0744 0.7906 0.5994 0.8667 0.0942 0.9042 0.6866 0.741 0.2913 0.0679 0.7003 0.3441 0.0592 0 0.5347 0.6239 0.3023 0.2741 0.1624 0.4883 6 0.5689 0.9926 0.6139",
    "6354 1372.6841 0.909 0.7293 0.0763 0.4364 0.3931 0.7504 0.8776 0.483 0.0082 0.7969 0.6211 0.7593 0.1878 0.2366 0.8595 0.3085 0.1442 0 0.3566 0.5427 0.117 0.1678 0.2511 0.4029 4 0.4176 0.9061 0.4554",
    "4096 872.3001 0.9099 0.8326 0.23 0.567 0.1134 0.8159 0.6325 0.9128 0.0497 0.9865 0.7885 0.6477 0.3407 0.3569 0.4903 0.4698 0.0492 0 0.7237 0.42 0.3913 0.4025 0.1555 0.2765 6 0.2723 0.794 0.5521",
    "3105 1588.0394 0.9033 0.6285 0.6155 0.7569 0.4354 0.776 0.3053 0.8485 0.0675 0.9962 0.9036 0.4581 0.3176 0.2053 0.2094 0.5299 0.1651 0 0.8513 0.3264 0.367 0.1892 0.0889 0.4491 7 0.3047 0.9235 0.403"
]

import sys
count = 0;

reproduccion2 = [];
for r in reproduccion:
    aux = [];
    r2 = r.split();
    for r3 in r2:
        aux.append(r3);
    reproduccion2.append(aux.copy());

parameters = ["destruct", "temp_inicial", "alpha", "prob_CambiarPrimarios", "prob_CambiarSecundarios", "prob_MoverPaciente_bloque", "prob_MoverPaciente_dia", "prob_EliminarPaciente", "prob_AgregarPaciente_1", "prob_AgregarPaciente_2",
              "prob_DestruirAgregar10", "prob_MejorarAfinidad_primario", "prob_MejorarAfinidad_secundario", "prob_AdelantarDia", "prob_MejorOR", "prob_AdelantarTodos", "prob_CambiarPaciente1", "prob_CambiarPaciente2", "prob_CambiarPaciente3",
              "destruct_type", "prob_DestruirOR", "prob_elite", "prob_GRASP", "prob_normal", "prob_Busq", "GRASP_alpha", "elite_size", "prob_GRASP1", "prob_GRASP2", "prob_GRASP3"];

dict_configs = {};
count = 0;
for p in parameters:
    dict_configs[p] = [];
    for r in reproduccion2:
        dict_configs[p].append(r[count]);
    count += 1;

group_i = ["prob_CambiarPrimarios", "prob_CambiarSecundarios", "prob_MoverPaciente_bloque", "prob_MoverPaciente_dia", 
           "prob_EliminarPaciente", "prob_AgregarPaciente_1", "prob_AgregarPaciente_2", "prob_DestruirAgregar10"];
group_ii = ["prob_MejorarAfinidad_primario", "prob_MejorarAfinidad_secundario", "prob_AdelantarDia", "prob_MejorOR", 
            "prob_AdelantarTodos", "prob_CambiarPaciente1", "prob_CambiarPaciente2", "prob_CambiarPaciente3"];
group_iii = ["prob_DestruirOR", "prob_elite", "prob_GRASP", "prob_normal"];
group_iv = ["prob_GRASP1", "prob_GRASP2", "prob_GRASP3"];

df_temp = pd.DataFrame(dict_configs);
for col in df_temp.columns:
    if col not in ["Instancias"]:
        df_temp[col] = pd.to_numeric(df_temp[col], errors="coerce");

def normalize_group(df, group_cols):
    df[group_cols] = df[group_cols].astype(float);
    group_values = df[group_cols].values;
    row_sums = group_values.sum(axis=1).reshape(-1, 1);
    row_sums[row_sums == 0] = 1;
    normalized = group_values / row_sums;
    df[group_cols] = normalized;
    return df;

for group in [group_i, group_ii, group_iii, group_iv]:
    df_temp = normalize_group(df_temp, group);
#df_temp.set_index("Instancias", inplace=True);
df_out = df_temp.T.reset_index();
df_out.rename(columns={"index": "Parámetro"}, inplace=True);
#df_out.to_csv("reproduccion.csv", index=False, sep=";", float_format="%.4f");

dict_gaps = {"Ejecuciones": [f"Ejec{i}" for i in range(len(reproduccion))]};
for i in range(1, 16):
    dict_gaps[f"Instancia {i}"] = [];
    count = 0;
    for r in reproduccion:
        count += 1;
        valores = r.split();
        sys.argv = ["metaheuristic.py", "0", "0", "0", f"../irace/instances/instance{i}.json"] + valores;
        print(f"Reproducción {count}, instancia {i}:", end=" ");
        result = metaheuristic.main();
        dict_gaps[f"Instancia {i}"].append(result);

df_gaps = pd.DataFrame(dict_gaps);
with pd.ExcelWriter("reproduccion.xlsx", engine="xlsxwriter") as writer:
    df_out.to_excel(writer, sheet_name="Parámetros", index=False, float_format="%.4f");
    df_gaps.to_excel(writer, sheet_name="Gaps", index=False, float_format="%.4f");