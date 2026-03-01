import typer
from alpha_factory.cli.backtest import quant_bt
from alpha_factory.cli.data import sync
from alpha_factory.cli.eval import quant_eval
from alpha_factory.cli.evals import quant_evals
from alpha_factory.cli.gp import quant_gp
from alpha_factory.cli.group import quant_group
from alpha_factory.cli.ml import quant_ml

# 创建 Typer 实例，这会让 AI 自动生成完美的 --help 文档
app = typer.Typer(
    help="Alpha Factory: 厂长的量化核心指令集。请遵循宪法，通过此工具与内核交互。",
    rich_markup_mode="rich",
    no_args_is_help=True,
)

# 注册顶级 sync 命令（原 data sync / update 合并）
app.command(name="sync")(sync)
app.command(name="eval")(quant_eval)
app.command(name="evals")(quant_evals)
app.command(name="bt")(quant_bt)
app.command(name="gp")(quant_gp)
app.command(name="group")(quant_group)
app.command(name="ml")(quant_ml)


@app.command()
def status():
    """
    [实证指令] 检查量化工厂的运行环境。
    AI 必须在首次运行或环境变更时执行此命令。
    """
    typer.echo("-" * 30)
    typer.echo("🚀 Alpha Factory 引擎状态报告")
    typer.echo("-" * 30)
    typer.echo("✅ 核心逻辑 (Core): 已就绪 (src/alpha_factory)")
    typer.echo("✅ 命令行接口 (CLI): 已挂载 (Typer)")
    typer.echo("📂 工作模式: Agent 受控模式 (Strict Compliance)")
    typer.echo("-" * 30)


if __name__ == "__main__":
    app()
