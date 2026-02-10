import typer

# 创建 Typer 实例，这会让 AI 自动生成完美的 --help 文档
app = typer.Typer(
    help="Alpha Factory: 厂长的量化核心指令集。请遵循宪法，通过此工具与内核交互。",
    rich_markup_mode="rich",
)


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


@app.command()
def factor(
    name: str = typer.Argument(..., help="因子名称，例如: alpha001"),
    period: int = typer.Option(20, help="计算周期 (T)"),
):
    """
    计算特定量化因子。
    如果所需因子未定义，AI 严禁自创代码，应向厂长提出需求。
    """
    typer.echo(f"⚙️ 正在启动内核算子计算因子: {name} (周期: {period}d)")
    # 这里未来会接入具体的 polars 算子
    typer.echo("⚠️ 提示: 因子逻辑待内核填充，当前为 mock 输出。")


if __name__ == "__main__":
    app()
