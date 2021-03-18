from pathlib import Path


class BacktestDataManager:

    def __init__(self) -> None:
        """Constructor"""
        self.backtest_result_path = Path(__file__).parent.parent.joinpath("backtest_result")
