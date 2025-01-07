import random

def CambiarPrimarios(sol: tuple, OT, hablar=False) -> tuple:
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    p1, p2 = random.sample([i for i, paciente in enumerate(pacientes) if paciente != -1], 2);
    t1, t2 = pacientes[p1], pacientes[p2];
    cir1, cir2 = primarios[t1], primarios[t2];
    dur1, dur2 = int(OT[p1]), int(OT[p2]);
    print(f"Moviendo cirujano primario {cir1} desde {t1} hasta {t2}...") if hablar else None;
    for t in range(dur1):
        primarios[t1 + t] = cir2;
    print(f"Moviendo cirujano primario {cir2} desde {t2} hasta {t1}...") if hablar else None;
    for t in range(dur2):
        primarios[t2 + t] = cir1;
    print("Primarios intercambiados.") if hablar else None;
    return (pacientes, primarios, secundarios)

def CambiarSecundarios(sol: tuple, OT, hablar=False) -> tuple:
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    p1, p2 = random.sample([i for i, paciente in enumerate(pacientes) if paciente != -1], 2);
    t1, t2 = pacientes[p1], pacientes[p2];
    cir1, cir2 = secundarios[t1], secundarios[t2];
    dur1, dur2 = int(OT[p1]), int(OT[p2]);
    print(f"Moviendo cirujano secundario {cir1} desde {t1} hasta {t2}...") if hablar else None;
    for t in range(dur1):
        secundarios[t1 + t] = cir2;
    print(f"Moviendo cirujano secundario {cir2} desde {t2} hasta {t1}...") if hablar else None;
    for t in range(dur2):
        secundarios[t2 + t] = cir1;
    print("Secundarios intercambiados.") if hablar else None;
    return (pacientes, primarios, secundarios)

def MoverPaciente_bloque(sol: tuple, OT, nSlot, nDays, hablar=False) -> tuple:
    def compress(o, d, t):
        return o * nSlot * nDays + d * nSlot + t

    def decompress(val):
        o = (val) // (nSlot * nDays);
        temp = (val) % (nSlot * nDays);
        d = temp // nSlot;
        t = temp % nSlot;
        return o, d, t
    pac_aux, prim_aux, sec_aux = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    p = random.sample([i for i, paciente in enumerate(pacientes) if paciente != -1], 1)[0];
    prim = primarios[pacientes[p]];
    sec = secundarios[pacientes[p]];
    o, d, t = decompress(pacientes[p]);
    dur = int(OT[p]);
    mov = random.choice([-1, 1]);
    if t + mov >= 1 and t + mov < nSlot - dur:
        if mov == -1:
            if pacientes[p] - 1 in primarios:
                print(f"Paciente {p} (duración {dur}) no pudo moverse (bloque {t}).") if hablar else None;
                return (pac_aux, prim_aux, sec_aux)
            else:
                del primarios[pacientes[p] + dur - 1];
                del secundarios[pacientes[p] + dur - 1];
                primarios[pacientes[p] - 1] = prim;
                secundarios[pacientes[p] - 1] = sec;
                pacientes[p] = compress(o, d, t+mov);
                print(f"Paciente {p} (duración {dur}) movido de bloque {t} a bloque {t+mov}.") if hablar else None;
                return (pacientes, primarios, secundarios)
        else:
            if pacientes[p] + dur in primarios:
                print(f"Paciente {p} (duración {dur}) no pudo moverse (bloque {t}).") if hablar else None;
                return (pac_aux, prim_aux, sec_aux)
            else:
                del primarios[pacientes[p]];
                del secundarios[pacientes[p]];
                primarios[pacientes[p] + dur] = prim;
                secundarios[pacientes[p] + dur] = sec;
                pacientes[p] = compress(o, d, t+mov);
                print(f"Paciente {p} (duración {dur}) movido de bloque {t} a bloque {t+mov}.") if hablar else None;
                return (pacientes, primarios, secundarios)
    print(f"Paciente {p} (duración {dur}) no pudo moverse desde bloque {t} hasta bloque {t+mov}.") if hablar else None;
    return (pac_aux, prim_aux, sec_aux)

def MoverPaciente_dia(sol: tuple, OT, nSlot, nDays, hablar=False) -> tuple:
    def compress(o, d, t):
        return o * nSlot * nDays + d * nSlot + t

    def decompress(val):
        o = (val) // (nSlot * nDays);
        temp = (val) % (nSlot * nDays);
        d = temp // nSlot;
        t = temp % nSlot;
        return o, d, t
    
    pac_aux, prim_aux, sec_aux = sol[0].copy(), sol[1].copy(), sol[2].copy();
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    p = random.sample([i for i, paciente in enumerate(pacientes) if paciente != -1], 1)[0];
    prim = primarios[pacientes[p]];
    sec = secundarios[pacientes[p]];
    o, d, t = decompress(pacientes[p]);
    dur = int(OT[p]);
    mov = random.choice([-1, 1]);
    if d + mov >= 1 and d + mov < nDays:
        for b in range(dur):
            bloque = compress(o, d+mov, t+b)
            if bloque in primarios:
                print(f"Paciente {p} no pudo moverse desde día {d} hasta día {d+mov}.") if hablar else None;
                return (pac_aux, prim_aux, sec_aux)
            else:
                del primarios[compress(o, d, t+b)];
                primarios[bloque] = prim;
                del secundarios[compress(o, d, t+b)];
                secundarios[bloque] = sec;
        pacientes[p] = compress(o, d+mov, t);
        print(f"Paciente {p} movido desde día {d} hasta día {d+mov}.") if hablar else None;
        return (pacientes, primarios, secundarios)
    print(f"Paciente {p} no pudo moverse desde día {d} hasta día {d+mov}.") if hablar else None;
    return (pac_aux, prim_aux, sec_aux)

def EliminarPaciente(sol: tuple, OT, hablar=None) -> tuple:
    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    try:
        p = random.sample([i for i, paciente in enumerate(pacientes) if paciente != -1], 1)[0];
    except:
        print("No hay pacientes asignados.") if hablar else None;
        return (pacientes, primarios, secundarios)
    dur = int(OT[p]);
    for b in range(dur):
        del primarios[pacientes[p] + b];
        del secundarios[pacientes[p] + b];
    pacientes[p] = -1;
    print(f"Paciente {p} eliminado.") if hablar else None;
    return (pacientes, primarios, secundarios)

import random

def AgregarPaciente(sol: tuple, AOR, OT, nSlot, nDays, room, slot, day, surgeon, second, hablar=False) -> tuple:
    def compress(o, d, t):
        return o * nSlot * nDays + d * nSlot + t

    def decompress(val):
        o_ = val // (nSlot * nDays)
        temp = val % (nSlot * nDays)
        d_ = temp // nSlot
        t_ = temp % nSlot
        return o_, d_, t_

    pacientes, primarios, secundarios = sol[0].copy(), sol[1].copy(), sol[2].copy();
    candidatos = [i for i, paciente in enumerate(pacientes) if paciente == -1];
    if not candidatos:
        print("No hay pacientes para agregar.") if hablar else None;
        return (pacientes, primarios, secundarios)

    p = random.choice(candidatos);
    dur = int(OT[p]);
    espacios_disponibles = [];
    for quir in room:
        for d_ in day:
            for t_ in slot:
                if t_ + dur - 1 >= len(slot):
                    continue
                esta_disponible = True
                for b in range(dur):
                    bloque = compress(quir, d_, t_ + b)
                    if AOR[p][quir][t_ + b][d_ % 5] != 1:
                        esta_disponible = False
                        break
                    if bloque in primarios or bloque in secundarios:
                        esta_disponible = False
                        break
                if esta_disponible:
                    inicio_bloque = compress(quir, d_, t_)
                    espacios_disponibles.append(inicio_bloque)
    if not espacios_disponibles:
        print("No hay espacio para asignar.") if hablar else None;
        return (pacientes, primarios, secundarios)
    bloque_inicial = random.choice(espacios_disponibles)
    quir_asign, dia_asign, slot_asign = decompress(bloque_inicial)

    if not surgeon or not second:
        print("No hay cirujanos disponibles en las listas.") if hablar else None;
        return (pacientes, primarios, secundarios)

    pacientes[p] = bloque_inicial;
    for b in range(dur):
        bloque = bloque_inicial + b
        cir_primario = random.choice(surgeon)
        cir_secundario = random.choice(second)
        primarios[bloque] = cir_primario
        secundarios[bloque] = cir_secundario

    print(f"Paciente {p} asignado. "+f"Quirófano={quir_asign}, Día={dia_asign}, Slot={slot_asign}, Duración={dur}.") if hablar else None;
    return (pacientes, primarios, secundarios)