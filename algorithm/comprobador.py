import sys, importlib, pandas as pd, os
import meta_test
importlib.reload(meta_test)

reproduccion = ['--destruct 193 --prob_CambiarPrimarios 0.3465 --prob_CambiarSecundarios 0.303 --prob_MoverPaciente_bloque 0.5731 --prob_MoverPaciente_dia 0.0971 --prob_EliminarPaciente 0.7342 --prob_AgregarPaciente_1 0.4583 --prob_AgregarPaciente_2 0.8911 --prob_DestruirAgregar10 0.0629 --prob_MejorarAfinidad_primario 0.2595 --prob_MejorarAfinidad_secundario 0.3745 --prob_AdelantarDia 0.2255 --prob_MejorOR 0.6648 --prob_AdelantarTodos 0.5087 --prob_CambiarPaciente1 0.6032 --prob_CambiarPaciente2 0.0776 --prob_CambiarPaciente3 0.5125 --prob_CambiarPaciente4 0.7596 --destruct_type 1 --prob_DestruirOR 0.364 --prob_elite 0.4122 --prob_GRASP 0.8552 --prob_normal 0.4001 --prob_Busq 0.9207 --GRASP_alpha 0.3469 --elite_size 3 --prob_GRASP1 0.8474 --prob_GRASP2 0.3635 --prob_GRASP3 0.3885 --acceptance_criterion No']
NUM_FLAGS = [
    "--destruct","--temp_ini","--alpha",
    "--prob_CambiarPrimarios","--prob_CambiarSecundarios",
    "--prob_MoverPaciente_bloque","--prob_MoverPaciente_dia",
    "--prob_EliminarPaciente","--prob_AgregarPaciente_1","--prob_AgregarPaciente_2",
    "--prob_DestruirAgregar10","--prob_DestruirAfinidad_Todos","--prob_DestruirAfinidad_Uno",
    "--prob_PeorOR", "--prob_AniquilarAfinidad",
    "--prob_MejorarAfinidad_primario","--prob_MejorarAfinidad_secundario",
    "--prob_AdelantarDia","--prob_MejorOR","--prob_AdelantarTodos",
    "--prob_CambiarPaciente1","--prob_CambiarPaciente2","--prob_CambiarPaciente3",
    "--prob_CambiarPaciente4","--prob_CambiarPaciente5",
    "--destruct_type","--prob_DestruirOR","--prob_elite","--prob_GRASP","--prob_normal",
    "--prob_Busq","--GRASP_alpha","--elite_size",
    "--prob_GRASP1","--prob_GRASP2","--prob_GRASP3"
]
CAT_FLAGS = ["--BusqTemp","--acceptance_criterion"]
ALL_FLAGS = NUM_FLAGS + CAT_FLAGS

DEFAULTS = {f:0             for f in NUM_FLAGS}
DEFAULTS.update({
    "--BusqTemp":"no",
    "--acceptance_criterion":"No"
})

def parse_line(line:str)->dict:
    toks = line.split()
    return {toks[i]:toks[i+1] for i in range(0,len(toks),2)}

parsed = [ {**DEFAULTS, **parse_line(l)} for l in reproduccion ]
df = pd.DataFrame(parsed)
df[NUM_FLAGS] = df[NUM_FLAGS].apply(pd.to_numeric, errors="coerce").fillna(0)

GROUP_I = [c for c in NUM_FLAGS if any(k in c for k in
    ["CambiarPrimarios","CambiarSecundarios","MoverPaciente","EliminarPaciente",
     "AgregarPaciente","DestruirAgregar10","DestruirAfinidad","PeorOR", "AniquilarAfinidad"])]
GROUP_II = [c for c in NUM_FLAGS if any(k in c for k in
    ["MejorarAfinidad","AdelantarDia","MejorOR","AdelantarTodos","CambiarPaciente"])]
GROUP_III = ["--prob_DestruirOR","--prob_elite","--prob_GRASP","--prob_normal"]
GROUP_IV = ["--prob_GRASP1","--prob_GRASP2","--prob_GRASP3"]

parsed_rows = [parse_line(l) for l in reproduccion]
df = pd.DataFrame(parsed_rows).reindex(columns=NUM_FLAGS).fillna(0).astype(float)

def _norm(cols):
    vals = df[cols].values
    totals = vals.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    df[cols] = vals / totals
for group in (GROUP_I, GROUP_II, GROUP_III, GROUP_IV):
    _norm(group)
df_out = (df.T.reset_index()
              .rename(columns={"index": "Parámetro"}))

METRIC_NAMES = ["Promedio","Mejor","Promedio_Gap","Mejor_Gap",
                "Tiempo","AvgIter","BestIter","NumSched"]

n_exec     = len(reproduccion)
cols_lvl1  = [f"Ejec{j+1}" for j in range(n_exec) for _ in METRIC_NAMES]
cols_lvl2  = METRIC_NAMES * n_exec
multi_cols = pd.MultiIndex.from_arrays([cols_lvl1, cols_lvl2])

df_gaps = pd.DataFrame(columns=multi_cols)
df_gaps.insert(0, ("", ""), []) 


def flush_to_excel(df_gaps_partial: pd.DataFrame) -> None:
    mode = "w" if not os.path.exists("reproduccion.xlsx") else "a"

    with pd.ExcelWriter("reproduccion.xlsx",
                        engine="openpyxl",
                        mode=mode,
                        if_sheet_exists="replace") as w:
        df_out.to_excel(w, sheet_name="Parámetros", float_format="%.4f")
        df_gaps_partial.to_excel(w, sheet_name="Gaps", float_format="%.4f")

flush_to_excel(df_gaps)

for inst in range(1, 6):
    df_gaps.loc[inst-1, ("", "")] = f"Instancia {inst}"
    for conf_idx, line in enumerate(reproduccion, start=1):
        argv = ["meta_test.py", "0", "0", "0", f"../irace/instances/instance{inst}.json"] + line.split()
        sys.argv = argv
        print(f"Instancia {inst:02d}  Ejec. {conf_idx:02d}")
        result = meta_test.main()
        start = (conf_idx - 1) * len(METRIC_NAMES)
        stop  = start + len(METRIC_NAMES)
        df_gaps.iloc[inst-1, 1+start : 1+stop] = result
        flush_to_excel(df_gaps)

print("✅ Todo terminado; resultados guardados en 'reproduccion.xlsx'")