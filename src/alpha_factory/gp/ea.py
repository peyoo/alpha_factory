import numpy as np
from deap import tools, algorithms
from loguru import logger


def eaMuPlusLambda_NSGA2(
    population,
    toolbox,
    mu,
    lambda_,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    early_stopping_rounds=15,
    delta=1e-4,
    generator=None,
):
    """
    专门为 NSGA-II 优化的进化循环
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # 1. 初始评估 (使用并行 map)
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(invalid_ind)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)

    best_valid_score = -np.inf
    stagnation_counter = 0

    # 2. 进化主循环
    for gen in range(1, ngen + 1):
        if generator and "independence" in generator.opt_names:
            # 传入最新的 HOF 对象，内部自动提取表达式并清理寄存站
            generator.dep_manager.update_and_prune(halloffame)

        # 变异与交叉
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # 评估后代
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        # 使用 set 计算唯一表达式数量
        # 注意：如果 ind 没有 expr_str 属性，请使用 str(ind)
        unique_exprs = set(str(ind) for ind in invalid_ind)

        unique_count = len(unique_exprs)
        total_count = len(invalid_ind)
        redundancy = (1 - unique_count / total_count) * 100 if total_count > 0 else 0

        logger.debug(f"Generation {gen}: Evaluating {total_count} individuals.")
        logger.debug(
            f"Unique Expressions: {unique_count} | Redundancy: {redundancy:.2f}%"
        )

        # 如果重复率过高，给出警告
        if redundancy > 50:
            logger.warning(
                f"High redundancy detected! Over {redundancy:.2f}% of offspring are clones."
            )

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # 选择下一代 (合并父代与子代进行非支配排序)
        population[:] = toolbox.select(population + offspring, mu)

        unique_pop = []
        seen = set()
        for ind in population:
            expr = str(ind)
            if expr not in seen:
                unique_pop.append(ind)
            seen.add(expr)

        logger.debug(
            f"Post-selection unique individuals: {len(unique_pop)} out of {len(population)}"
        )
        population[:] = unique_pop

        if halloffame is not None:
            halloffame.update(offspring)

            # 日志输出排在前5位的个体，及其表达式
            logger.debug("Hall of Fame Top 10 Individuals:")
            for i, ind in enumerate(halloffame.items[:10]):
                logger.debug(f"Rank {i + 1}: {ind} with fitness {ind.fitness.values}")

        # 统计
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # 3. 优化后的早停逻辑 (监控验证集指标提升)
        # 假设 stats 中记录了验证集的最高 ann_ret，存放在 record['max'] 的特定索引
        # 这里以第一个目标作为监控主指标
        current_score = record["max"][0] if "max" in record else 0

        if current_score > (best_valid_score + delta):
            best_valid_score = current_score
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        if stagnation_counter >= early_stopping_rounds:
            print(f"Early Stopping: 指标已连续 {early_stopping_rounds} 代无实质提升。")
            break

    return population, logbook
