import random

random.seed(10)


def play_game(strategy1, strategy2, rounds=200):
    score1, score2 = 0, 0
    history1, history2 = [], []
    flag1, flag2 = 0, 0
    for _ in range(rounds):
        move1 = strategy1(history2, flag2)
        move2 = strategy2(history1, flag1)
        if move1 == "betray":
            flag1 = 1
        if move2 == "betray":
            flag2 = 1
        history1.append(move1)
        history2.append(move2)

        if move1 == "cooperate" and move2 == "cooperate":
            score1 += 5
            score2 += 5
        elif move1 == "cooperate" and move2 == "betray":
            score1 += 0
            score2 += 10
        elif move1 == "betray" and move2 == "cooperate":
            score1 += 10
            score2 += 0
        elif move1 == "betray" and move2 == "betray":
            score1 += 1
            score2 += 1

    return score1, score2


# Strategies
def leifeng_strategy(history, flag):  # 雷锋策略
    return "cooperate"


def caocao_strategy(history, flag):  # 曹操策略
    return "betray"


def random_strategy(history, flag):  # 精神病患者策略
    return random.choice(["cooperate", "betray"])


def tit_for_tat_strategy(history, flag):  # 一报还一报策略
    if not history:
        return "cooperate"
    else:
        return history[-1]  # Return the opponent's last move


def cruel_strategy(history, flag):  # 冷酷策略
    if not history:
        return "cooperate"
    elif flag == 1:
        return "betray"
    else:
        return "cooperate"


# Play the game
# 雷锋策略 vs. 曹操策略
leifeng_score, caocao_score = play_game(leifeng_strategy, caocao_strategy)
print(f"雷锋策略VS.曹操策略\n雷锋策略玩家得分: {leifeng_score}, 曹操策略玩家得分: {caocao_score}\n")

# 雷锋策略 vs. 精神病策略
leifeng_score, random_score = play_game(leifeng_strategy, random_strategy)
print(f"雷锋策略VS.精神病策略\n雷锋策略玩家得分: {leifeng_score}, 精神病策略玩家得分: {random_score}\n")

# 雷锋策略 vs. 一报还一报策略
leifeng_score, tit_fortat_score = play_game(leifeng_strategy, tit_for_tat_strategy)
print(f"雷锋策略VS.一报还一报策略\n雷锋策略玩家得分: {leifeng_score}, 一报还一报策略玩家得分: {tit_fortat_score}\n")

# 曹操策略 vs. 精神病策略
caocao_score, random_score = play_game(caocao_strategy, random_strategy)
print(f"曹操策略VS.精神病策略\n曹操策略玩家得分: {caocao_score}, 精神病策略玩家得分: {random_score}\n")

# 曹操策略 vs. 一报还一报策略
caocao_score, tit_fortat_score = play_game(caocao_strategy, tit_for_tat_strategy)
print(f"曹操策略VS.一报还一报策略\n曹操策略玩家得分: {caocao_score}, 一报还一报策略玩家得分: {tit_fortat_score}\n")

# 精神病策略 vs. 一报还一报策略
random_score, tit_fortat_score = play_game(random_strategy, tit_for_tat_strategy)
print(f"精神病策略VS.一报还一报策略\n精神病策略玩家得分: {random_score}, 一报还一报策略玩家得分: {tit_fortat_score}\n")

# 雷锋策略 vs. 冷酷策略
leifeng_score, cruel_score = play_game(leifeng_strategy, cruel_strategy)
print(f"雷锋策略VS.冷酷策略\n雷锋策略玩家得分: {leifeng_score}, 冷酷策略玩家得分: {cruel_score}\n")

# 曹操策略 vs. 冷酷策略
caocao_score, cruel_score = play_game(caocao_strategy, cruel_strategy)
print(f"曹操策略VS.冷酷策略\n曹操策略玩家得分: {caocao_score}, 冷酷策略玩家得分: {cruel_score}\n")

# 精神病策略 vs. 冷酷策略
random_score, cruel_score = play_game(random_strategy, cruel_strategy)
print(f"精神病策略VS.冷酷策略\n精神病策略玩家得分: {random_score}, 冷酷策略玩家得分: {cruel_score}\n")

# 一报还一报策略 vs. 冷酷策略
tit_fortat_score, cruel_score = play_game(tit_for_tat_strategy, cruel_strategy)
print(f"一报还一报策略VS.冷酷策略\n一报还一报策略玩家得分: {tit_fortat_score}, 冷酷策略玩家得分: {cruel_score}\n")

# 冷酷策略 vs. 冷酷策略
cruel_score1, cruel_score2 = play_game(cruel_strategy, cruel_strategy)
print(f"冷酷策略VS.冷酷策略\n冷酷策略玩家1得分: {cruel_score1}, 冷酷策略玩家2得分: {cruel_score2}\n")

# 雷锋策略玩家1 vs. 雷锋策略玩家2
leifeng_score1, leifeng_score2 = play_game(leifeng_strategy, leifeng_strategy)
print(f"雷锋策略VS.雷锋策略\n雷锋策略玩家1得分: {leifeng_score1}, 雷锋策略玩家2得分: {leifeng_score2}\n")

# 精神病策略玩家1 vs. 精神病策略玩家2
random_score1, random_score2 = play_game(random_strategy, random_strategy)
print(f"精神病策略VS.精神病策略\n精神病策略玩家1得分: {random_score1}, 精神病策略玩家2得分: {random_score2}\n")

# 一报还一报策略玩家1 vs. 一报还一报策略玩家2
tit_for_tat_score1, tit_for_tat_score2 = play_game(tit_for_tat_strategy, tit_for_tat_strategy)
print(
    f"一报还一报策略VS.一报还一报策略\n一报还一报策略玩家1得分: {tit_for_tat_score1}, 一报还一报策略玩家2得分: {tit_for_tat_score2}\n")
