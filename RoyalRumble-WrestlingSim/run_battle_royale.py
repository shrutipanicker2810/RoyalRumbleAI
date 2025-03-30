from env import WrestlingEnv
from wrestler import Wrestler
from coach import Coach
from evolution_algo import Evolution
from run_match import run_match
import random

def run_battle_royale():
    env = WrestlingEnv()
    wrestlers_data = [
        ("John Cena", 85, 185, 114, 10),
        ("The Rock", 90, 196, 118, 10),
        ("The Undertaker", 95, 208, 136, 9),
        ("Triple H", 88, 193, 116, 9),
        ("Randy Orton", 82, 196, 113, 8),
        ("Brock Lesnar", 78, 191, 130, 9),
        ("Roman Reigns", 75, 191, 120, 9),
        ("Seth Rollins", 72, 185, 98, 8),
        ("Stone Cold Steve Austin", 92, 188, 114, 10),
        ("Hulk Hogan", 95, 201, 137, 9),
        ("Ric Flair", 98, 185, 110, 8),
        ("Shawn Michaels", 90, 185, 102, 8),
        ("Bret Hart", 85, 183, 106, 7),
        ("AJ Styles", 80, 180, 99, 8),
        ("Edge", 82, 193, 109, 8),
        ("Chris Jericho", 88, 183, 103, 8),
        ("Big Show", 85, 213, 200, 7),
        ("Kane", 82, 208, 147, 8),
        ("Rey Mysterio", 90, 168, 79, 9),
        ("Batista", 75, 198, 130, 8),
        ("CM Punk", 72, 185, 98, 7),
        ("Kevin Owens", 68, 183, 122, 7),
        ("Finn Bálor", 70, 180, 86, 7),
        ("Becky Lynch", 65, 168, 61, 8),
        ("Charlotte Flair", 65, 180, 72, 8),
        ("Ronda Rousey", 60, 168, 61, 8),
        ("Macho Man Randy Savage", 85, 188, 112, 7),
        ("André the Giant", 90, 224, 240, 9),
        ("Dwayne 'The Rock' Johnson", 90, 196, 118, 10),
        ("Chyna", 70, 178, 82, 7),
        ("Mick Foley", 85, 188, 130, 7),
        ("Bray Wyatt", 65, 191, 129, 7)
    ]
    wrestlers = [Wrestler(env, name, i, pop, height, weight, exp) for i, (name, pop, height, weight, exp) in enumerate(wrestlers_data)]
    random.shuffle(wrestlers)
    
    print("Battle Royale Entry Order and Initial Health (Random Order):")
    print("---------------------------------------------")
    for i, wrestler in enumerate(wrestlers, 1):
        print(f"{i}. {wrestler.name} - Initial Health: {wrestler.health}")
    print("---------------------------------------------")
    
    for i, w in enumerate(wrestlers):
        w.set_opponents([x for j, x in enumerate(wrestlers) if j != i])

    remaining_wrestlers = wrestlers.copy()
    eliminated_wrestlers = []
    current_wrestlers = [remaining_wrestlers.pop(0), remaining_wrestlers.pop(0)]
    match_results = []
    print(f"Initial match: {current_wrestlers[0].name} vs {current_wrestlers[1].name}")

    while len(remaining_wrestlers) > 0 or len(current_wrestlers) > 1:
        final_state = run_match(current_wrestlers[0], current_wrestlers[1], env, render=True)  # Enable rendering
        match_results.append(final_state)
        total_rewards = final_state["rewards"]
        health = final_state["health"]
        stamina = final_state["stamina"]

        if total_rewards[0] > total_rewards[1]:
            winner = current_wrestlers[0]
            loser = current_wrestlers[1]
        elif total_rewards[1] > total_rewards[0]:
            winner = current_wrestlers[1]
            loser = current_wrestlers[0]
        else:
            if health[0] > health[1]:
                winner = current_wrestlers[0]
                loser = current_wrestlers[1]
            elif health[1] > health[0]:
                winner = current_wrestlers[1]
                loser = current_wrestlers[0]
            else:
                winner = current_wrestlers[0] if stamina[0] >= stamina[1] else current_wrestlers[1]
                loser = current_wrestlers[1] if stamina[0] >= stamina[1] else current_wrestlers[0]

        eliminated_wrestlers.append(loser)
        print(f"\n{loser.name} is eliminated! {winner.name} remains in the ring with {winner.health} health.")
        print(f"Current Elimination Order: {[w.name for w in eliminated_wrestlers]}")

        if remaining_wrestlers:
            next_wrestler = remaining_wrestlers.pop(0)
            next_wrestler.reset(xyz=(-0.5, 0, 1.0))
            winner.reset(xyz=(0.5, 0, 1.0))
            current_wrestlers = [winner, next_wrestler]
            print(f"Next match: {winner.name} vs {next_wrestler.name}")
        else:
            current_wrestlers = [winner]

    final_winner = current_wrestlers[0]
    print(f"\nBattle Royale Winner: {final_winner.name}!")
    print(f"Elimination Order: {[w.name for w in eliminated_wrestlers]}")
    env.close()  # Close the viewer when done
    return match_results, wrestlers

if __name__ == "__main__":
    match_results, wrestlers = run_battle_royale()

    # Optimize pairings with Coach class
    coach = Coach(wrestlers,WrestlingEnv())
    best_pairing, best_strategies = coach.simulated_annealing()

    # Evolve wrestlers for next season
    evolution = Evolution(wrestlers)
    next_season_wrestlers = evolution.evolve()