import sys, importlib, pandas as pd, os
import meta_test
importlib.reload(meta_test)

reproduccion = [
    "--destruct 3486 --temp_ini 1628.627 --alpha 0.998 --prob_CambiarPrimarios 0.5072 --prob_CambiarSecundarios 0.5246 --prob_MoverPaciente_bloque 0.1069 --prob_MoverPaciente_dia 0.1599 --prob_EliminarPaciente 0.2362 --prob_AgregarPaciente_1 0.6621 --prob_AgregarPaciente_2 0.9674 --prob_DestruirAgregar10 0.4148 --prob_DestruirAfinidad_Todos 0.1435 --prob_DestruirAfinidad_Uno 0.6722 --prob_PeorOR 0.2447 --prob_AniquilarAfinidad 0.6051 --prob_MejorarAfinidad_primario 0.6632 --prob_MejorarAfinidad_secundario 0.7004 --prob_AdelantarDia 0.5502 --prob_MejorOR 0.0744 --prob_AdelantarTodos 0.4141 --prob_CambiarPaciente1 0.8321 --prob_CambiarPaciente2 0.253 --prob_CambiarPaciente3 0.1407 --prob_CambiarPaciente4 0.5862 --prob_CambiarPaciente5 0.3964 --destruct_type 1 --prob_DestruirOR 0.4114 --prob_elite 0.8667 --prob_GRASP 0.1738 --prob_normal 0.1347 --prob_Busq 0.9298 --BusqTemp yes --GRASP_alpha 0.4942 --elite_size 7 --prob_GRASP1 0.3496 --prob_GRASP2 0.399 --prob_GRASP3 0.4853 --acceptance_criterion SA --tabu 0 --ini_random 0.4307",
    "--destruct 5447 --temp_ini 1666.1383 --alpha 0.9994 --prob_CambiarPrimarios 0.7551 --prob_CambiarSecundarios 0.6494 --prob_MoverPaciente_bloque 0.0487 --prob_MoverPaciente_dia 0.0656 --prob_EliminarPaciente 0.5378 --prob_AgregarPaciente_1 0.7177 --prob_AgregarPaciente_2 0.6134 --prob_DestruirAgregar10 0.3514 --prob_DestruirAfinidad_Todos 0.0491 --prob_DestruirAfinidad_Uno 0.5821 --prob_PeorOR 0.4592 --prob_AniquilarAfinidad 0.5969 --prob_MejorarAfinidad_primario 0.7098 --prob_MejorarAfinidad_secundario 0.7457 --prob_AdelantarDia 0.2043 --prob_MejorOR 0.0518 --prob_AdelantarTodos 0.8827 --prob_CambiarPaciente1 0.5464 --prob_CambiarPaciente2 0.4653 --prob_CambiarPaciente3 0.544 --prob_CambiarPaciente4 0.6218 --prob_CambiarPaciente5 0.0202 --destruct_type 1 --prob_DestruirOR 0.7406 --prob_elite 0.4872 --prob_GRASP 0.4661 --prob_normal 0.2294 --prob_Busq 0.7537 --BusqTemp yes --GRASP_alpha 0.1062 --elite_size 10 --prob_GRASP1 0.4783 --prob_GRASP2 0.6519 --prob_GRASP3 0.7646 --acceptance_criterion SA --tabu 0 --ini_random 0.5115",
    "--destruct 6531 --temp_ini 1672.1167 --alpha 0.9985 --prob_CambiarPrimarios 0.9732 --prob_CambiarSecundarios 0.5839 --prob_MoverPaciente_bloque 0.2507 --prob_MoverPaciente_dia 0.1148 --prob_EliminarPaciente 0.3981 --prob_AgregarPaciente_1 0.6225 --prob_AgregarPaciente_2 0.5321 --prob_DestruirAgregar10 0.3545 --prob_DestruirAfinidad_Todos 0.2475 --prob_DestruirAfinidad_Uno 0.7546 --prob_PeorOR 0.2615 --prob_AniquilarAfinidad 0.7977 --prob_MejorarAfinidad_primario 0.6718 --prob_MejorarAfinidad_secundario 0.7889 --prob_AdelantarDia 0.2701 --prob_MejorOR 0.1631 --prob_AdelantarTodos 0.7989 --prob_CambiarPaciente1 0.477 --prob_CambiarPaciente2 0.3921 --prob_CambiarPaciente3 0.7201 --prob_CambiarPaciente4 0.2891 --prob_CambiarPaciente5 0.1604 --destruct_type 1 --prob_DestruirOR 0.6787 --prob_elite 0.3088 --prob_GRASP 0.5659 --prob_normal 0.2904 --prob_Busq 0.9566 --BusqTemp yes --GRASP_alpha 0.4753 --elite_size 9 --prob_GRASP1 0.3319 --prob_GRASP2 0.4681 --prob_GRASP3 0.4314 --acceptance_criterion SA --tabu 0 --ini_random 0.4443",
    "--destruct 5597 --temp_ini 1500.2097 --alpha 0.9992 --prob_CambiarPrimarios 0.6633 --prob_CambiarSecundarios 0.5498 --prob_MoverPaciente_bloque 0.2046 --prob_MoverPaciente_dia 0.148 --prob_EliminarPaciente 0.3298 --prob_AgregarPaciente_1 0.7205 --prob_AgregarPaciente_2 0.9418 --prob_DestruirAgregar10 0.265 --prob_DestruirAfinidad_Todos 0.288 --prob_DestruirAfinidad_Uno 0.7908 --prob_PeorOR 0.3051 --prob_AniquilarAfinidad 0.7521 --prob_MejorarAfinidad_primario 0.6712 --prob_MejorarAfinidad_secundario 0.7567 --prob_AdelantarDia 0.5438 --prob_MejorOR 0.2684 --prob_AdelantarTodos 0.4762 --prob_CambiarPaciente1 0.6921 --prob_CambiarPaciente2 0.3716 --prob_CambiarPaciente3 0.059 --prob_CambiarPaciente4 0.6538 --prob_CambiarPaciente5 0.4406 --destruct_type 1 --prob_DestruirOR 0.5975 --prob_elite 0.8704 --prob_GRASP 0.2242 --prob_normal 0.1668 --prob_Busq 0.9988 --BusqTemp yes --GRASP_alpha 0.4512 --elite_size 8 --prob_GRASP1 0.5425 --prob_GRASP2 0.317 --prob_GRASP3 0.3916 --acceptance_criterion SA --tabu 0 --ini_random 0.4928",
    "--destruct 5236 --temp_ini 1734.8052 --alpha 0.9996 --prob_CambiarPrimarios 0.5915 --prob_CambiarSecundarios 0.894 --prob_MoverPaciente_bloque 0.5861 --prob_MoverPaciente_dia 0.6123 --prob_EliminarPaciente 0.5069 --prob_AgregarPaciente_1 0.4185 --prob_AgregarPaciente_2 0.8941 --prob_DestruirAgregar10 0.1524 --prob_DestruirAfinidad_Todos 0.8368 --prob_DestruirAfinidad_Uno 0.5947 --prob_PeorOR 0.1888 --prob_AniquilarAfinidad 0.841 --prob_MejorarAfinidad_primario 0.2226 --prob_MejorarAfinidad_secundario 0.708 --prob_AdelantarDia 0.6446 --prob_MejorOR 0.9943 --prob_AdelantarTodos 0.309 --prob_CambiarPaciente1 0.6753 --prob_CambiarPaciente2 0.0342 --prob_CambiarPaciente3 0.0758 --prob_CambiarPaciente4 0.8658 --prob_CambiarPaciente5 0.4865 --destruct_type 1 --prob_DestruirOR 0.7296 --prob_elite 0.7156 --prob_GRASP 0.7391 --prob_normal 0.1865 --prob_Busq 0.8476 --BusqTemp no --GRASP_alpha 0.2701 --elite_size 6 --prob_GRASP1 0.3724 --prob_GRASP2 0.0875 --prob_GRASP3 0.4868 --acceptance_criterion SA --tabu 0 --ini_random 0.4593",
    "--destruct 6208 --temp_ini 1529.0067 --alpha 0.9959 --prob_CambiarPrimarios 0.6584 --prob_CambiarSecundarios 0.6702 --prob_MoverPaciente_bloque 0.1485 --prob_MoverPaciente_dia 0.192 --prob_EliminarPaciente 0.1177 --prob_AgregarPaciente_1 0.687 --prob_AgregarPaciente_2 0.7227 --prob_DestruirAgregar10 0.4811 --prob_DestruirAfinidad_Todos 0.2949 --prob_DestruirAfinidad_Uno 0.8585 --prob_PeorOR 0.2026 --prob_AniquilarAfinidad 0.8899 --prob_MejorarAfinidad_primario 0.6614 --prob_MejorarAfinidad_secundario 0.7636 --prob_AdelantarDia 0.2964 --prob_MejorOR 0.4499 --prob_AdelantarTodos 0.5427 --prob_CambiarPaciente1 0.7811 --prob_CambiarPaciente2 0.1747 --prob_CambiarPaciente3 0.2894 --prob_CambiarPaciente4 0.5791 --prob_CambiarPaciente5 0.3886 --destruct_type 1 --prob_DestruirOR 0.6032 --prob_elite 0.6033 --prob_GRASP 0.3763 --prob_normal 0.2634 --prob_Busq 0.9243 --BusqTemp yes --GRASP_alpha 0.4888 --elite_size 10 --prob_GRASP1 0.333 --prob_GRASP2 0.3551 --prob_GRASP3 0.3487 --acceptance_criterion SA --tabu 0 --ini_random 0.4361",
    "--destruct 6914 --temp_ini 1565.8221 --alpha 0.9989 --prob_CambiarPrimarios 0.8797 --prob_CambiarSecundarios 0.384 --prob_MoverPaciente_bloque 0.0959 --prob_MoverPaciente_dia 0.1265 --prob_EliminarPaciente 0.2293 --prob_AgregarPaciente_1 0.7388 --prob_AgregarPaciente_2 0.5704 --prob_DestruirAgregar10 0.3971 --prob_DestruirAfinidad_Todos 0.3763 --prob_DestruirAfinidad_Uno 0.5561 --prob_PeorOR 0.1204 --prob_AniquilarAfinidad 0.7767 --prob_MejorarAfinidad_primario 0.555 --prob_MejorarAfinidad_secundario 0.9704 --prob_AdelantarDia 0.1751 --prob_MejorOR 0.5352 --prob_AdelantarTodos 0.6174 --prob_CambiarPaciente1 0.8051 --prob_CambiarPaciente2 0.4984 --prob_CambiarPaciente3 0.3893 --prob_CambiarPaciente4 0.6015 --prob_CambiarPaciente5 0.4211 --destruct_type 1 --prob_DestruirOR 0.7588 --prob_elite 0.6493 --prob_GRASP 0.2129 --prob_normal 0.5006 --prob_Busq 0.9519 --BusqTemp yes --GRASP_alpha 0.3666 --elite_size 9 --prob_GRASP1 0.068 --prob_GRASP2 0.1961 --prob_GRASP3 0.2927 --acceptance_criterion SA --tabu 0 --ini_random 0.3487"
]

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
    "--prob_Busq", "--ils_extra", "--GRASP_alpha","--elite_size",
    "--prob_GRASP1","--prob_GRASP2","--prob_GRASP3",
    "--tabu", "--tabulen", "--ini_random"
]
CAT_FLAGS = ["--BusqTemp","--acceptance_criterion"]
ALL_FLAGS = NUM_FLAGS + CAT_FLAGS

DEFAULTS = {f:0 for f in NUM_FLAGS}
DEFAULTS.update({
    "--BusqTemp":"yes",
    "--acceptance_criterion":"SA"})

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
GROUP_III = ["--prob_DestruirOR","--prob_elite","--prob_GRASP","--prob_normal", "--ini_random"]
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
    file_exists = os.path.exists("reproduccionMayo.xlsx");
    kwargs = dict(engine="openpyxl", mode="a" if file_exists else "w");
    if file_exists:
        kwargs["if_sheet_exists"] = "replace";
    with pd.ExcelWriter("reproduccionMayo.xlsx", **kwargs) as w:
        df_out.to_excel(w, sheet_name="Parámetros", float_format="%.4f");
        df_gaps_partial.to_excel(w, sheet_name="Gaps", float_format="%.4f");

flush_to_excel(df_gaps)

for inst in range(1, 16):
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

print("✅ Todo terminado; resultados guardados en 'reproduccionMayo.xlsx'")

# /opt/homebrew/Cellar/python@3.10/3.10.17/Frameworks/Python.framework/Versions/3.10/bin/python3.10 comprobador.py