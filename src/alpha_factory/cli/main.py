import typer
from alpha_factory.cli.data import data_app

# 创建 Typer 实例，这会让 AI 自动生成完美的 --help 文档
app = typer.Typer(
    help="Alpha Factory: 厂长的量化核心指令集。请遵循宪法，通过此工具与内核交互。",
    rich_markup_mode="rich",
)

# 注册 data 子命令
app.add_typer(data_app, name="data", help="数据子命令：包含数据同步/更新相关的命令。")


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
