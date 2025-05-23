#!/usr/bin/env python3
import sys, warnings, json, pandas as pd, random, numpy as np, time, copy, math, importlib, argparse
from pathlib import Path
class CSVCheckpoint:
    def __init__(self, secs, csv_path, instance, aggregator=None):
        self.secs = sorted(secs)
        self.targetfile = Path(csv_path)
        self.instance = instance
        self.next_idx = 0
        self.first = not self.targetfile.exists()
        self.targetfile.parent.mkdir(parents=True, exist_ok=True)
        self.gaps = []

        self.best_solution = None
        self.best_gap = float('inf')
        self.no_improvement_count = 0
        self.max_no_improvement = 1000
        self.total_iterations = 0        # never resets
        self.iter_best_global = 0 
        self.aggregator = aggregator

    def notify(self, elapsed, gap, patients):
        self.gaps.append(gap)
        self.total_iterations += 1

        if gap < self.best_gap:
            self.best_gap = gap
            self.iter_best_global = self.total_iterations
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.next_idx >= len(self.secs) or elapsed < self.secs[self.next_idx]:
            return

        avg_gap = sum(self.gaps) / len(self.gaps)

        row_best = self.best_gap
        row_avg  = avg_gap
        row_iter = self.total_iterations
        row_iter_best = self.iter_best_global

        if self.aggregator is not None:
            # forward to aggregator instead of writing
            self.aggregator.add(self.next_idx,
                                row_best, row_avg,
                                row_iter, row_iter_best,
                                patients)
        else:
            pd.DataFrame([{
                "instance": self.instance,
                "time": elapsed,
                "best_gap": row_best,
                "avg_gap":  row_avg,
                "iterations": row_iter,
                "iter_best": row_iter_best,
                "patients": patients
            }]).to_csv(
                self.targetfile,
                mode="a",
                index=False,
                header=self.first
            )
            self.first = False
        self.next_idx += 1

    def should_restart(self):
        """Return True if the no‑improvement counter reached its limit."""
        return self.no_improvement_count >= self.max_no_improvement

    def reset(self):
        """Reset the no‑improvement counter after a restart."""
        self.no_improvement_count = 0
class CSVCheckpointAggregator:
    """Aggregate checkpoint stats across many runs, then write averaged rows."""
    def __init__(self, secs, csv_path, instance):
        self.secs       = secs
        self.targetfile = Path(csv_path)
        self.instance   = instance
        self.data       = [dict(best_gaps=[], avg_gaps=[], iterations=[], iter_bests=[], patients=[]) for _ in secs]
        self.first      = not self.targetfile.exists()

    def add(self, idx, best_gap, avg_gap, iterations, iter_best, patients):
        self.data[idx]["best_gaps"].append(best_gap)
        self.data[idx]["avg_gaps"].append(avg_gap)
        self.data[idx]["iterations"].append(iterations)
        self.data[idx]["iter_bests"].append(iter_best)
        self.data[idx]["patients"].append(patients)

    def finalize(self):
        rows = []
        for idx, sec in enumerate(self.secs):
            bucket = self.data[idx]
            if not bucket["best_gaps"]:
                continue
            avg_gap      = sum(bucket["avg_gaps"]) / len(bucket["avg_gaps"])
            iterations   = int(sum(bucket["iterations"]) / len(bucket["iterations"]))
            best_gap     = min(bucket["best_gaps"])
            best_idx     = bucket["best_gaps"].index(best_gap)
            iter_best    = bucket["iter_bests"][best_idx]
            patients   = int(sum(bucket["patients"]) / len(bucket["patients"]))
            rows.append({
                "instance": self.instance,
                "time": sec,
                "best_gap": best_gap,
                "avg_gap":  avg_gap,
                "iterations": iterations,
                "iter_best": iter_best,
                "patients": patients
            })
        if rows:
            pd.DataFrame(rows).to_csv(
                self.targetfile,
                mode="a",
                index=False,
                header=self.first
            )
            self.first = False


warnings.filterwarnings("ignore")

import perturbations
importlib.reload(perturbations)
from perturbations import (
    CambiarPrimarios, CambiarSecundarios, MoverPaciente_bloque, MoverPaciente_dia,
    EliminarPaciente, AgregarPaciente_1, AgregarPaciente_2, DestruirAgregar10,
    DestruirAfinidad_Todos, DestruirAfinidad_Uno, PeorOR, AniquilarAfinidad
)

import localsearches as localsearches
importlib.reload(localsearches)
from localsearches import (
    MejorarAfinidad_primario, MejorarAfinidad_secundario, AdelantarDia, MejorOR,
    AdelantarTodos, CambiarPaciente1, CambiarPaciente2, CambiarPaciente3,
    CambiarPaciente4, CambiarPaciente5
)

import initial_solutions as initial_solutions
importlib.reload(initial_solutions)
from initial_solutions import normal, GRASP, complete_random

from evaluation import EvalAllORs 

testing=False; parametroFichas=0.11; entrada="etapa1"; version="C"; solucion_inicial=True

PERTURBATIONS=[
    CambiarPrimarios, CambiarSecundarios, MoverPaciente_bloque, MoverPaciente_dia,
    EliminarPaciente, AgregarPaciente_1, AgregarPaciente_2, DestruirAgregar10,
    DestruirAfinidad_Todos, DestruirAfinidad_Uno, PeorOR, AniquilarAfinidad
]

LOCAL_SEARCHES=[
    MejorarAfinidad_primario, MejorarAfinidad_secundario, AdelantarDia, MejorOR,
    AdelantarTodos, CambiarPaciente1, CambiarPaciente2, CambiarPaciente3,
    CambiarPaciente4, CambiarPaciente5
]

def compress(o,d,t): 
    return o*nSlot*nDays+d*nSlot+t
def decompress(val):
    o=val//(nSlot*nDays); 
    temp=val%(nSlot*nDays); 
    d=temp//nSlot; 
    t=temp%nSlot
    return o,d,t
def WhichExtra(o,t,d,e): return int(extras[o][t][d%5][e])

def load_data_and_config():
    global dfSurgeon, dfSecond, dfRoom, dfType, dfPatient
    global dfdisORA, dfdisORB, dfdisORC, dfdisORD
    global dfdisAffi, dfdisAffiDiario, dfdisAffiBloque, dfdisRank
    global dfExtra, dfRankExtra, extras, num_ext
    global dfPatient
    global patient, surgeon, second, room, day, nSlot, nRooms, nDays, slot, T, nFichas
    global level_affinity, OT, AOR, COIN, Ex, I, SP
    global dictCosts

    #if testing == False:
    #    with open(f"input/{entrada}.json") as file:
    #        data = json.load(file)
    #else:
    #    with open("input/inst_test.json") as file:
    #        data = json.load(file)

    #config = data["configurations"]
    dfSurgeon = pd.read_excel("../input/MAIN_SURGEONS.xlsx", sheet_name='surgeon', converters={'n°': int}, index_col=[0])
    dfSecond = pd.read_excel("../input/SECOND_SURGEONS.xlsx", sheet_name='second', converters={'n°': int}, index_col=[0])
    dfRoom = pd.read_excel("../input/ROOMS.xlsx", sheet_name='room', converters={'n°': int}, index_col=[0])
    dfType = pd.read_excel("../input/PROCESS_TYPE.xls", sheet_name='Process Type', converters={'n°': int}, index_col=[0])

    if typePatients == "low":
        dfPatient = pd.read_csv("../input/LowPriority.csv");
    elif typePatients == "high":
        dfPatient = pd.read_csv("../input/HighPriority.csv");
    else:
        dfPatient = pd.read_csv("../input/AllPriority.csv");
    
    dfPatient = dfPatient.iloc[:nPatients];

    extras = []
    def load_OR_dist(sheet_name):
        df_ = pd.read_excel("../input/DIST_OR_EXT.xlsx", sheet_name=sheet_name, converters={'n°':int}, index_col=[0])
        df_ = df_.astype(str).values.tolist()
        ext_ = []
        for i in range(len(df_)):
            aux = df_[i].copy()
            ext_.append(aux)
        for i in range(len(df_[0])):
            part = df_[0][i].split(";")
            df_[0][i] = part[0]
            ext_[0][i] = part[1:]
            part2 = df_[1][i].split(";")
            df_[1][i] = part2[0]
            ext_[1][i] = part2[1:]
        return df_, ext_

    dfdisORA, extraA = load_OR_dist('A')
    dfdisORB, extraB = load_OR_dist('B')
    dfdisORC, extraC = load_OR_dist('C')
    dfdisORD, extraD = load_OR_dist('D')
    extras.append(extraA)
    extras.append(extraB)
    extras.append(extraC)
    extras.append(extraD)

    # Affinity sheets
    dfdisAffi = pd.read_excel("../input/AFFINITY_EXT.xlsx", sheet_name='Hoja1', converters={'n°':int}, index_col=[0])
    dfdisAffiDiario = pd.read_excel("../input/AFFINITY_DIARIO.xlsx", sheet_name='Dias', converters={'n°':int}, index_col=[0])
    dfdisAffiBloque = pd.read_excel("../input/AFFINITY_DIARIO.xlsx", sheet_name='Bloques', converters={'n°':int}, index_col=[0])
    dfdisRank = pd.read_excel("../input/RANKING.xlsx", sheet_name='Hoja1', converters={'n°':int}, index_col=[0])

    num_ext = 2;

    extra = [];
    dfExtra = [];
    for i in range(num_ext):
        aux = pd.read_excel("../input/AFFINITY_EXT.xlsx", sheet_name='Extra'+str(i+1), converters = {'n°':int}, index_col=[0]);
        extra.append(len(aux));
        dfExtra.append(aux);

    # Rankings for extras
    dfRankExtra = []
    for i in range(num_ext):
        aux = pd.read_excel("../input/RANKING.xlsx", sheet_name='Extra'+str(i+1), converters={'n°':int}, index_col=[0])
        dfRankExtra.append(aux)

    patient = [p for p in range(nPatients)];
    surgeon = [s for s in range(nSurgeons)];
    second = [a for a in range(nSurgeons)]
    room = [o for o in range(len(dfRoom))]
    day = [d for d in range(nDays)];
    nSlot = 16;  # Bloques de 30 minutos
    nRooms = len(room);
    slot = [t for t in range(nSlot)]
    T = nSlot//2; # División entre mañana y tarde

    I = np.ones((nPatients, nDays), dtype=int);
    for p in patient:
        for d in day:
            I[(p,d)] = 1 + dfPatient.iloc[p]["espera"] * dfPatient.iloc[p]["edad"] * 0.0001/(d+1);

    # Rankings para extras
    dfRankExtra = [];
    for i in range(num_ext):
        aux = pd.read_excel("../input/RANKING.xlsx", sheet_name='Extra'+str(i+1), converters = {'n°':int}, index_col=[0]);
        extra.append(len(aux));
        dfRankExtra.append(aux);

    # Matriz de coincidencia cirujanos
    COIN = np.zeros((nSurgeons, nSurgeons), dtype=int);
    for s in surgeon:
        for a in second:
            if dfSurgeon.iloc[s][0] == dfSecond.iloc[a][0]:
                COIN[(s,a)] = 1;
            else:
                COIN[(s,a)] = 0;
                

    Ex = [np.ones((nSurgeons, extra[i]), dtype=float) for i in range(num_ext)];
    for i in range(num_ext):
        for s in surgeon:
            for e in range(extra[i]):
                Ex[i][(s,e)] = dfExtra[i].iloc[e][s+1];

    dictCosts = {};

    for s in surgeon:
        for a in second:
            for _ in range(nSlot * nDays * len(room)):
                o, d, t = decompress(_);
                dictCosts[(s, a, _)] = int(dfdisAffi.iloc[a][s+1] + dfdisAffiDiario.iloc[d%5][s+1] + sum(Ex[i][(s,WhichExtra(o,t//T,d,i)-1)] for i in range(num_ext)) + dfdisAffiBloque.iloc[t//T][s+1]);
    
    DISP = np.ones(nPatients);
    
    process=[t for t in range(len(dfType))] # ?
    def busquedaType(especialidad):
        indice = 0;
        for i in range(len(process)):
            if (especialidad == dfType.iloc[i][0]):
                indice = i;
        return indice
    
    # Compatibilidad paciente-cirujano
    SP = np.zeros((nPatients, nSurgeons), dtype=int);
    contador = 0;
    for p in patient:
        for s in surgeon:
            if busquedaType(dfPatient.iloc[p]["especialidad"]) == busquedaType(dfSurgeon.iloc[s][9]):
                SP[p][s] = 1;       

    OR_specialties = {};
    for o in range(len(room)):
        for d in range(5):
            for t in range(2):
                if o == 0:
                    aux = dfdisORA[t][d];
                elif o == 1:
                    aux = dfdisORB[t][d];
                elif o == 2:
                    aux = dfdisORC[t][d];
                elif o == 3:
                    aux = dfdisORD[t][d];
                OR_specialties[(o, d, t)] = aux;

    # Diccionario de paciente
    dic_p = {p: [0, 0, 0, 0, 0] for p in patient};
    for p in patient:
        #dic_p[p][0] = list_patient[p] #paciente y número aleatorio asociado (entre 0 y 1)
        dic_p[p][1] = busquedaType(dfPatient.iloc[p]["especialidad"]) # ID Especialidad
        dic_p[p][2] = dfPatient.iloc[p]["nombre"]; #Nombre del paciente
        dic_p[p][3] = p; # ID
        dic_p[p][4] = dfPatient.iloc[p]["especialidad"]; # Especialidad requerida

    AOR = np.zeros((nPatients, nRooms, nSlot, 5));
    dicOR = {o:[] for o in room};
    j = [];
    z = [];
    ns = 0;
    for o in room:
        if o == 0:
            for d in range(5):
                for e in range(2):
                    if e == 0:                    
                        for t in range(len(slot)//2):

                            j = dfdisORA[e][d%5];
                            j = j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d%5,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):

                            j=dfdisORA[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d%5,t,z]
                                ns+=1
        if o==1:
            for d in range(5):
                for e in range(2):
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):
                            j=dfdisORB[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):
                            j=dfdisORB[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
        if o==2:
            for d in range(5):
                for e in range(2):
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):
                            j=dfdisORC[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):
                            j=dfdisORC[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
        if o==3:
            for d in range(5):
                for e in range(2):
                    if e==0:                    
                        for t in range(0,int(len(slot)/2)):
                            j=dfdisORD[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1
                    if e==1:
                        for t in range(int(len(slot)/2),len(slot)):
                            j=dfdisORD[e][d%5]
                            j=j.split("#")
                            for a in range(len(j)):
                                z=j[a]
                                dicOR[ns]=[o,d,t,z]
                                ns+=1

    p = 0;
    o = 0;
    t = 0;

    for ns in range(len(dic_p)):
        for nP in range(len(dicOR)):
            if str(dic_p[ns][1])==dicOR[nP][3]:
                p = dic_p[ns][3];
                o = dicOR[nP][0];
                t = dicOR[nP][2];
                d = dicOR[nP][1];
                AOR[p][o][t][d%5] = 1;

    level_affinity = min_affinity;
    OT = np.zeros(nPatients, dtype=int);
    for p in patient:
        OT[p] = int(dfPatient.iloc[p]["tipo"]);
    nFichas = int((parametroFichas * 4 * nSlot * len(room) * 2 * 3 )/(len(surgeon)**0.5));

def weighted_choice(items, weights):
    pool = [(item, w) for item, w in zip(items, weights) if w > 0]
    if not pool:
        return random.choice(items)
    total = sum(w for _, w in pool)
    r = random.uniform(0, total)
    upto = 0
    for item, w in pool:
        upto += w
        if upto >= r:
            return item

def sa(initial_solution, temp_inicial, alpha, pert_probs, ls_probs, seed, report_secs, time_limit, listener=None, iterations=None):
    random.seed(seed)
    current = copy.deepcopy(initial_solution)
    current_cost = EvalAllORs(current[0], VERSION=version,
                            hablar=False,
                            nFichas_val=nFichas,
                            day_py=day,
                            surgeon_py=surgeon,
                            room_py=room,
                            OT_obj=OT,
                            I_obj=I,
                            dictCosts_obj=dictCosts,
                            nDays_val=nDays,
                            nSlot_val=nSlot,
                            SP_obj=SP,
                            bks=bks)

    best = copy.deepcopy(current)
    best_cost = current_cost
    temp = temp_inicial
    init_time = time.time()
    rep = sorted(report_secs)
    nxt = 0
    it = 0
    if listener and listener.best_solution is None:
        listener.best_solution = copy.deepcopy(current)

    while True:
        it += 1
        neighbour = copy.deepcopy(current)
        try:
            pert_func = weighted_choice(PERTURBATIONS, pert_probs)
            result = pert_func(neighbour, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays)
            if result is not None:
                neighbour = result
        except Exception as e:
            neighbour = copy.deepcopy(current)
        try:
            ls_func = weighted_choice(LOCAL_SEARCHES, ls_probs)
            result = ls_func(neighbour, surgeon, second, OT, I, SP, AOR, dictCosts, nSlot, nDays)
            if result is not None:
                neighbour = result
        except Exception:
            # keep neighbour unchanged if LS fails
            pass
        try:
            neigh_cost = EvalAllORs(neighbour[0],VERSION=version,
                                    hablar=False,
                                    nFichas_val=nFichas,
                                    day_py=day,
                                    surgeon_py=surgeon,
                                    room_py=room,
                                    OT_obj=OT,
                                    I_obj=I,
                                    dictCosts_obj=dictCosts,
                                    nDays_val=nDays,
                                    nSlot_val=nSlot,
                                    SP_obj=SP,
                                    bks=bks)
        except Exception:
            # fall back to a very bad cost so the neighbour is rejected
            neigh_cost = float('inf')

        
        if neigh_cost-current_cost<0 or random.random()<math.exp(-(neigh_cost-current_cost)/temp):
            current, current_cost = copy.deepcopy(neighbour), neigh_cost
            if neigh_cost < best_cost:
                best, best_cost = copy.deepcopy(neighbour), neigh_cost
                if listener:
                    listener.best_solution = copy.deepcopy(best)

        temp *= alpha
        elapsed = time.time()-init_time
        
        if listener:
            patients_scheduled = sum(1 for p in best[0][0] if p != -1)
            listener.notify(elapsed, best_cost, patients_scheduled)
            if listener.should_restart():
                current = copy.deepcopy(listener.best_solution)
                current_cost = EvalAllORs(current[0], VERSION=version,
                                        hablar=False,
                                        nFichas_val=nFichas,
                                        day_py=day,
                                        surgeon_py=surgeon,
                                        room_py=room,
                                        OT_obj=OT,
                                        I_obj=I,
                                        dictCosts_obj=dictCosts,
                                        nDays_val=nDays,
                                        nSlot_val=nSlot,
                                        SP_obj=SP,
                                        bks=bks)
                temp = temp_inicial
                listener.reset()

        if elapsed >= time_limit:
            break
        if nxt < len(rep) and elapsed >= rep[nxt]:
            nxt += 1
        if nxt >= len(rep):
            break
        if temp < 1e-6:
            temp = temp_inicial

    return best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations",type=int,default=5000)
    parser.add_argument("--temp_inicial",type=float,default=800.0)
    parser.add_argument("--alpha",type=float,default=0.99)
    parser.add_argument("--seed",type=int,default=258)
    for p in ["CambiarPrimarios","CambiarSecundarios","MoverPaciente_bloque","MoverPaciente_dia",
              "EliminarPaciente","AgregarPaciente_1","AgregarPaciente_2","DestruirAgregar10",
              "DestruirAfinidad_Todos","DestruirAfinidad_Uno","PeorOR","AniquilarAfinidad"]:
        parser.add_argument(f"--prob_{p}",type=float,default=10.0)
    for s in ["MejorarAfinidad_primario","MejorarAfinidad_secundario","AdelantarDia","MejorOR",
              "AdelantarTodos","CambiarPaciente1","CambiarPaciente2","CambiarPaciente3",
              "CambiarPaciente4","CambiarPaciente5"]:
        parser.add_argument(f"--prob_{s}",type=float,default=10.0)
    parser.add_argument("--report_minutes", type=str, default="");
    args, _ = parser.parse_known_args();
    report_secs = [float(x)*60 for x in args.report_minutes.split(",") if x.strip()] if args.report_minutes.strip() else [];

    seeds = list(range(10))
    for i in range(1, 16):
        try:
            aggregator = CSVCheckpointAggregator(report_secs, "sa_checkpoints.csv", f"instance{i}")
            with open(f"../irace/instances/instance{i}.json",'r') as f:
                data = json.load(f)

            global typePatients,nPatients,nDays,nSurgeons,bks,nFichas,min_affinity,day,slot
            typePatients = data["patients"]
            nPatients = int(data["n_patients"])
            nDays = int(data["days"])
            nSurgeons = int(data["surgeons"])
            bks = int(data["bks"])
            nFichas = int(data["fichas"])
            min_affinity = int(data["min_affinity"])
            time_limit = int(data["time_limit"])

            load_data_and_config()
            initial = GRASP(surgeon,second,patient,room,day,slot,AOR,I,dictCosts,nFichas,nSlot,
                          SP,COIN,OT,alpha=0.2,modo=1,VERSION="C",hablar=False)

            if initial is None:
                print(f"Warning: Could not generate initial solution for instance {i}, skipping...")
                continue

            pert_probs = [getattr(args,f"prob_{p}") for p in ["CambiarPrimarios","CambiarSecundarios","MoverPaciente_bloque","MoverPaciente_dia",
                      "EliminarPaciente","AgregarPaciente_1","AgregarPaciente_2","DestruirAgregar10",
                      "DestruirAfinidad_Todos","DestruirAfinidad_Uno","PeorOR","AniquilarAfinidad"]]
            ls_probs = [getattr(args,f"prob_{s}") for s in ["MejorarAfinidad_primario","MejorarAfinidad_secundario","AdelantarDia","MejorOR",
                      "AdelantarTodos","CambiarPaciente1","CambiarPaciente2","CambiarPaciente3",
                      "CambiarPaciente4","CambiarPaciente5"]]

            solutions = []
            for ejec in seeds:
                checkpoint_listener = CSVCheckpoint(
                    report_secs,
                    "sa_checkpoints.csv",
                    f"instance{i}",
                    aggregator=aggregator)
                best = sa(initial,
                          args.temp_inicial,
                          args.alpha,
                          pert_probs,
                          ls_probs,
                          ejec,                # seed per run
                          report_secs=report_secs,
                          time_limit=time_limit,
                          listener=checkpoint_listener,
                          iterations=args.iterations if not report_secs else None)
                solutions.append(EvalAllORs(best[0],VERSION=version,
                                    hablar=False,
                                    nFichas_val=nFichas,
                                    day_py=day,
                                    surgeon_py=surgeon,
                                    room_py=room,
                                    OT_obj=OT,
                                    I_obj=I,
                                    dictCosts_obj=dictCosts,
                                    nDays_val=nDays,
                                    nSlot_val=nSlot,
                                    SP_obj=SP,
                                    bks=bks))
            aggregator.finalize()
            print(f"Instance {i}: mean_cost = {-np.mean(solutions):.5f}, "
                  f"mean_gap = {1 - (-np.mean(solutions)/bks):.5f}")
            # print(f"Instance {i}: {EvalAllORs(best[0], VERSION='C')}")
        except Exception as e:
            print(f"Error processing instance {i}: {str(e)}")
            continue

if __name__=="__main__":
    main()

#python sa.py --temp_inicial 800.0 --alpha 0.99 --seed 258 --report_minutes "0.1,0.3,0.5" --iterations 500000 --prob_CambiarPrimarios 15.0 --prob_MejorOR 20.0