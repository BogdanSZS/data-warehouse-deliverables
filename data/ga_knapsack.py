#!/usr/bin/env python3
# =========================================================================================
#  Proiect: GA pe Knapsack 0/1 – comparație de variante (generațional, steady-state, memetic)
#
#  Ce sunt algoritmii genetici (GA)?
#  - Heuristici inspirate de evoluția biologică. Lucrează cu o populație de soluții (cromozomi),
#    pe care le selectează, recombină (crossover) și mută (mutation) pentru a îmbunătăți treptat
#    funcția obiectiv. Sunt buni pentru probleme cu spații mari/complexe, unde căutările exacte
#    sunt prea scumpe.
#
#  Problema Knapsack 0/1 (rucsacul):
#  - Avem n obiecte, fiecare cu valoare v_i și greutate w_i. Dorim să maximizăm suma valorilor
#    selectate, respectând o capacitate totală (sumă de greutăți) ≤ C. Reprezentăm o soluție ca
#    vector binar x ∈ {0,1}^n: x_i=1 dacă alegem obiectul i, altfel 0.
#
#  De ce GA pe Knapsack?
#  - Knapsack 0/1 este NP-hard și are peisaje de căutare cu multe platouri. GA-urile sunt o alegere
#    populară pentru astfel de probleme combinatoriale. Aici comparăm 3 scheme:
#    (1) GA generațional (cu elitism=1) – clasic, sincron, populație nouă per generație.
#    (2) GA steady-state – actualizări incrementale: înlocuiește pe rând cei mai slabi cu copii mai buni.
#    (3) GA memetic – GA + o trecere de căutare locală (hill-climbing 1-bit) pe elite (exploatare mai bună).
#
#  Design decizii importante:
#  - Reprezentare: binară (0/1) – naturală pentru Knapsack 0/1.
#  - Fezabilitate: folosim un operator de "reparare greedy" (dacă depășim capacitatea, scoatem itemii
#    cu cel mai slab raport valoare/greutate) – asta ajută GA-urile să rămână cât mai des în zona fezabilă.
#  - Fitness: valoare totală dacă soluția e fezabilă; altfel penalizare liniară (coeficient=10.0).
#  - Măsurători: mediem pe 3 rulări (seeds 101/202/303). Salvăm CSV + două grafice.
#
#  Output-uri:
#    - data/ga_knapsack_results.csv     – tabel: variantă, run, seed, best_value, duration_s
#    - figs/ga_knapsack_convergence.png – convergență (best-so-far mediu vs #evaluări)
#    - figs/ga_knapsack_bars.png        – bar chart cu medie ± deviație standard (zoom pe diferențe mici)
#
#  Comanda pentru cod: 
# python ga_knapsack.py \
#   --items 120 \
#   --gen-seed 2025 \
#   --max-evals 50000 \
#   --runs 3 \
#   --seeds 101 202 303 \
#   --out-data data/ga_knapsack_results.csv \
#   --fig-conv figs/ga_knapsack_convergence.png \
#   --fig-bar figs/ga_knapsack_bars.png
# =========================================================================================

import argparse, random, time
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean
from pathlib import Path

#-------------Problem-----------------
def make_knapsack(n_items: int, seed: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generează o instanță random de Knapsack 0/1.
    - n_items: numărul de obiecte
    - seed: seed determinist pentru reproducibilitate
    Returnează:
      values  (np.ndarray shape [n]): valorile v_i
      weights (np.ndarray shape [n]): greutățile w_i
      capacity (float): capacitatea totală C (aici 50% din suma greutăților)
    """
    rng = random.Random(seed)
    weights = np.array([rng.uniform(1.0, 20.0) for _ in range(n_items)])
    values = np.array([rng.uniform(5.0, 100.0) for _ in range(n_items)])
    capacity = 0.5 * weights.sum()
    return values, weights, capacity

def repair_greedy(ind: np.ndarray, w: np.ndarray, v: np.ndarray, cap: float):
    """
    Operator de reparare: dacă soluția depășește capacitatea, scoatem pe rând
    itemii cu cel mai slab raport valoare/greutate până când devine fezabilă.
    - ind: vector binar al soluției curente (în loc)
    """
    # dacă depășim capacitatea, scoatem itemii cu cel mai slab raport value/weight
    while (w @ ind) > cap:
        ones = np.where(ind == 1)[0]
        if len(ones) == 0:
            break
        ratios = v[ones] / w[ones]
        worst_idx = ones[np.argmin(ratios)]
        ind[worst_idx] = 0
        
def fitness(ind: np.ndarray, v: np.ndarray, w: np.ndarray, cap: float) -> float:
    """
    Funcția fitness: maximizează valoarea totală sub constrângerea de greutate.
    - Dacă e fezabilă: returnează suma valorilor.
    - Dacă e nefezabilă: penalizează liniar depășirea (coeficient=10.0).
      (coeficientul poate fi ajustat; prea mic => soluții nefezabile, prea mare => conservator)
    """
    W, V = float(w @ ind), float(v @ ind)
    if W <= cap: return V
    return V - 10.0 * (W - cap) # penalizare liniară

def one_point_crossover(a: np.ndarray, b: np.ndarray, rng: random.Random):
    """
    Crossover 1-punct: taie cromozomii într-un punct și schimbă segmentele.
    Returnează doi copii (copii independente de părinți).
    """
    if len(a) <= 1:
        return a.copy(), b.copy()
    cx = rng.randint(1, len(a)-1)
    c1 = np.concatenate([a[:cx], b[cx:]])
    c2 = np.concatenate([b[:cx], a[cx:]])
    return c1, c2

def bitflip_mutation(ind: np.ndarray, pm: float, rng: random.Random):
    """
    Mutație bit-flip: pentru fiecare genă, întoarce bitul cu probabilitatea pm.
    pm tipic pentru binar: ≈ 1/n (unde n = număr de gene).
    """
    for i in range(len(ind)):
        if rng.random() < pm:
            ind[i] ^= 1

def tournament_selection(pop: List[np.ndarray], fits: List[float], k: int, rng: random.Random):
    """
    Selecție prin turneu de mărime k: alege aleator k indivizi și întoarce cel mai bun.
    Avantaj: simplu, control fin al presiunii selecției prin k.
    """
    best_idx = rng.randrange(len(pop))
    best_fit = fits[best_idx]
    for _ in range(k-1):
        i = rng.randrange(len(pop))
        if fits[i] > best_fit:
            best_idx, best_fit = i, fits[i]
    return pop[best_idx].copy()

def local_search_1bit(ind: np.ndarray, v: np.ndarray, w: np.ndarray, cap: float):
    """
    Căutare locală simplă (memetic): încearcă să inverseze fiecare bit o dată (single pass).
    Acceptă imediat orice îmbunătățire. Întoarce cea mai bună variantă întâlnită.
    """
    best = ind.copy()
    best_f = fitness(best, v, w, cap)
    for i in range(len(ind)):
        cand = best.copy()
        cand[i] ^= 1
        f = fitness(cand, v, w, cap)
        if f > best_f:
            best, best_f = cand, f
    return best

#-------------GA-----------------
@dataclass
class GAConfig:
    """
    Set de hiper-parametri pentru rularea GA.
    - name: denumire prietenoasă pentru raportare/grafice
    - pop_size: mărimea populației
    - p_cx: probabilitatea de crossover
    - p_mut: probabilitatea de mutație per genă (≈ 1/n pentru codificare binară)
    - tour_k: mărimea turneului la selecție
    - elitism: câți indivizi de top copiem direct în generația nouă (numai la 'generational'/'memetic')
    - max_evals: bugetul total de evaluări ale fitness-ului (criteriu de oprire)
    - variant: 'generational' | 'steady' | 'memetic'
    - memetic_elite: câți dintre cei mai buni din noua populație primesc local search (numai la 'memetic')
    """
    name: str
    pop_size: int = 120
    p_cx: float = 0.9
    p_mut: float = 1.0/120.0
    tour_k: int = 3
    elitism: int = 1
    max_evals: int = 50_000
    variant: str = "generational" # generational | steady | memetic
    memetic_elite: int = 4

def run_ga(cfg: GAConfig, seed: int, v: np.ndarray, w: np.ndarray, cap: float) -> Dict[str, List[float]]:
    """
    Rulează o variantă GA conform configurării:
      - initializează populația random (densitate 10% de 1), apoi aplică reparare greață (greedy)
      - folosește selecție prin turneu, crossover 1-punct, mutație bit-flip
      - respectă schema: generational / steady / memetic
      - contorizează evaluările și reține istoricul celui mai bun "best-so-far"
    Returnează dict cu istoricul (best_hist, eval_hist), best_fit final și durata.
    """
    rng = random.Random(seed)
    n = len(v)
    pop = [np.array([1 if rng.random()<0.1 else 0 for _ in range(n)], dtype=int) for _ in range(cfg.pop_size)]
    for ind in pop: repair_greedy(ind, w, v, cap)
    fits = [fitness(ind, v, w, cap) for ind in pop]
    evals = len(pop)
    
    best_hist = [max(fits)]
    eval_hist = [evals]
    start = time.time()
    
    while evals < cfg.max_evals:
        if cfg.variant == "generational":
            new_pop = []
            # elitism
            elite_idx = np.argsort(fits)[-cfg.elitism:][::-1]
            for idx in elite_idx:
                new_pop.append(pop[idx].copy())
            #reproducere
            while len(new_pop) < cfg.pop_size:
                p1 = tournament_selection(pop, fits, cfg.tour_k, rng)
                p2 = tournament_selection(pop, fits, cfg.tour_k, rng)
                c1, c2 = (p1.copy(), p2.copy())
                if rng.random() < cfg.p_cx:
                    c1, c2 = one_point_crossover(p1, p2, rng)
                bitflip_mutation(c1, cfg.p_mut, rng)
                bitflip_mutation(c2, cfg.p_mut, rng)
                repair_greedy(c1, w, v, cap)
                repair_greedy(c2, w, v, cap)
                new_pop.append(c1)
                if len(new_pop) < cfg.pop_size: new_pop.append(c2)
            pop = new_pop
            fits = [fitness(ind, v, w, cap) for ind in pop]
            evals += len(pop)
        
        elif cfg.variant == "steady":
            for _ in range(cfg.pop_size // 2):
                p1 = tournament_selection(pop, fits, cfg.tour_k, rng)
                p2 = tournament_selection(pop, fits, cfg.tour_k, rng)
                c1, c2 = (p1.copy(), p2.copy())
                if rng.random() < cfg.p_cx:
                    c1, c2 = one_point_crossover(p1, p2, rng)
                bitflip_mutation(c1, cfg.p_mut, rng)
                bitflip_mutation(c2, cfg.p_mut, rng)
                repair_greedy(c1, w, v, cap)
                repair_greedy(c2, w, v, cap)
                f1 = fitness(c1, v, w, cap); f2 = fitness(c2, v, w, cap)
                evals += 2
                # înlocuiește cei mai slabi dacă noii copii sunt mai buni
                for cand, f in [(c1, f1), (c2, f2)]:
                    worst = int(np.argmin(fits))
                    if f > fits[worst]:
                        pop[worst] = cand; fits[worst] = f
        
        elif cfg.variant == "memetic":
            new_pop = []
            elite_idx = np.argsort(fits)[-cfg.elitism:][::-1]
            for idx in elite_idx: new_pop.append(pop[idx].copy())
            while len(new_pop) < cfg.pop_size:
                p1 = tournament_selection(pop, fits, cfg.tour_k, rng)
                p2 = tournament_selection(pop, fits, cfg.tour_k, rng)
                c1, c2 = (p1.copy(), p2.copy())
                if rng.random() < cfg.p_cx:
                    c1, c2 = one_point_crossover(p1, p2, rng)
                bitflip_mutation(c1, cfg.p_mut, rng)
                bitflip_mutation(c2, cfg.p_mut, rng)
                repair_greedy(c1, w, v, cap)
                repair_greedy(c2, w, v, cap)
                new_pop.append(c1)
                if len(new_pop) < cfg.pop_size: new_pop.append(c2)
            new_fits = [fitness(ind, v, w, cap) for ind in new_pop]
            evals += len(new_pop)
            # local search pe top-k
            topk = np.argsort(new_fits)[-cfg.memetic_elite:]
            for idx in topk:
                imp = local_search_1bit(new_pop[idx], v, w, cap)
                new_pop[idx] = imp
                new_fits[idx] = fitness(imp, v, w, cap)
                evals += 1
            pop, fits = new_pop, new_fits
        
        else:
            raise ValueError("variantă de algoritm necunoscută")
        
        best_hist.append(max(fits))
        eval_hist.append(evals)
    
    duration = time.time() - start
    return {"best_hist": best_hist, "eval_hist": eval_hist, "best_fit": max(fits), "duration_s": duration}

def main():
    """
    Punctul de intrare:
      - parsează argumentele CLI,
      - generează instanța de Knapsack,
      - rulează fiecare variantă GA pe 3 seed-uri,
      - salvează CSV + grafice (convergență și bar chart cu SD),
      - toate fișierele ies în subfolderele data/ și figs/.
    """
    p = argparse.ArgumentParser(description="GA vs GA pe Knapsack 0/1")
    p.add_argument("--items", type=int, default=120)
    p.add_argument("--gen-seed", type=int, default=2025, help="seed pentru generarea instanței knapsack")
    p.add_argument("--max-evals", type=int, default=50_000)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--seeds", type=int, nargs="*", default=[101,202,303])
    p.add_argument("--out-data", type=str, default="data/ga_knapsack_results.csv")
    p.add_argument("--fig-conv", type=str, default="figs/ga_knapsack_convergence.png")
    p.add_argument("--fig-bar", type=str, default="figs/ga_knapsack_bars.png")
    args = p.parse_args()
    
    Path("data").mkdir(exist_ok=True); Path("figs").mkdir(exist_ok=True)
    
    values, weights, capacity = make_knapsack(args.items, args.gen_seed)
    
    variants = [
        GAConfig(name="GA generational (elitism=1)", variant="generational", elitism=1, max_evals=args.max_evals),
        GAConfig(name="GA steady-state", variant="steady", max_evals=args.max_evals),
        GAConfig(name="Memetic GA (LS top-4)", variant="memetic", memetic_elite=4, max_evals=args.max_evals),
    ]
    
    rows = []
    curves = {}
    interp_x = np.linspace(0, args.max_evals, 200)
    
    for cfg in variants:
        ys = []; bests = []; durs = []
        for run_idx, s in enumerate(args.seeds[:args.runs]):
            out = run_ga(cfg, seed=int(s), v=values, w=weights, cap=capacity)
            rows.append({"variant": cfg.name, "run": run_idx+1, "seed": int(s),
                         "best_value": out["best_fit"], "duration_s": out["duration_s"]})
            x = np.array(out["eval_hist"], dtype=float)
            y = np.array(out["best_hist"], dtype=float)
            ys.append(np.interp(interp_x, x, y))
            bests.append(out["best_fit"]); durs.append(out["duration_s"])
        curves[cfg.name] = {"x": interp_x,
                            "y_mean": np.mean(np.vstack(ys), axis=0),
                            "best_mean": mean(bests),
                            "dur_mean_s": mean(durs)}
        
    df = pd.DataFrame(rows)
    df.to_csv(args.out_data, index=False)
    print(f"[OK] CSV salvat la: {args.out_data}")
    
    # Convergență
    plt.figure()
    for name, d in curves.items():
        plt.plot(d["x"], d["y_mean"], label=name)
    plt.xlabel("Evaluări de fitness")
    plt.ylabel("Cel mai bun scor (medie pe rulări)")
    plt.title(f"Convergență - Knapsack 0/1 (n={args.items})")
    plt.legend()
    plt.savefig(args.fig_conv, bbox_inches="tight"); plt.close()
    print(f"[OK] Convergență: {args.fig_conv}")
    
    # Bar chart (zoom + etichete + eroare std)
    names = list(curves.keys())

    # calculează media și deviația standard direct din df (pe aceleași nume/ordine)
    g = df.groupby("variant")["best_value"].agg(["mean", "std"]).reindex(names)
    means = g["mean"].tolist()
    errs  = g["std"].fillna(0).tolist()

    fig, ax = plt.subplots()
    x = range(len(names))
    bars = ax.bar(x, means, yerr=errs, capsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("Cel mai bun scor mediu (± Standard Deviation)")
    ax.set_title("Comparație variante GA")

    # zoom pe axa Y ca să vezi diferențele mici (~0.1%)
    mmin, mmax = min(means), max(means)
    delta = max(1.0, (mmax - mmin))
    pad = max(1.0, 0.8 * delta)
    ax.set_ylim(mmin - pad, mmax + pad)

    # etichete numerice deasupra barelor (2 zecimale)
    ax.bar_label(bars, fmt="%.2f", padding=3)

    fig.tight_layout()
    fig.savefig(args.fig_bar, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Bar chart: {args.fig_bar}")
    
if __name__ == "__main__":
    main()