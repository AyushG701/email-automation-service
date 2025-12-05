from enum import Enum
from dataclasses import dataclass
from typing import Optional
import logging


class LocationRisk(Enum):
    """Enumeration for different location risk levels"""
    HOT_MARKET = (
        1.10, "Hot Market - Major hub with instant load availability")
    STANDARD = (1.15, "Standard - Normal areas with decent load availability")
    HIGH_RISK = (1.25, "High Risk - Remote areas or middle of nowhere")

    def __init__(self, multiplier: float, description: str):
        self.multiplier = multiplier
        self.description = description


@dataclass
class BidResult:
    """Container for bid calculation results"""
    loaded_miles: int
    safety_multiplier: float
    estimated_total_miles: float
    cost_per_mile: float
    break_even_cost: float
    target_profit_margin: float
    calculated_bid: float
    min_rate: float
    max_rate: float
    # expected_profit: float
    # profit_margin_percent: float

#     def __str__(self) -> str:
#         return f"""
# === Bid Calculation Results ===
# Loaded Miles: {self.loaded_miles}
# Safety Multiplier: {self.safety_multiplier}x
# Estimated Total Miles: {self.estimated_total_miles:.0f}
# Estimated Deadhead: {self.estimated_total_miles - self.loaded_miles:.0f} miles

# Cost per Mile: ${self.cost_per_mile:.2f}
# Break-Even Cost: ${self.break_even_cost:.2f}
# Target Profit Margin: {self.target_profit_margin * 100:.0f}%

# Calculated Bid: ${self.calculated_bid:.2f}
# Max Bid (Rounded): ${self.max_rate:.2f}
# Min Bid (Rounded): ${self.min_rate:.2f}
# ================================
# """


class FreightBidCalculator:
    """
    Calculator for freight bidding using the Safety Multiplier Method.

    This class helps calculate competitive bids by accounting for deadhead miles
    using a safety multiplier based on location risk.
    """

    def __init__(self, cost_per_mile: float = 2.00, default_profit_margin: float = 0.20, max_profit_margin: float = 0.50):
        """
        Initialize the calculator with default values.

        Args:
            cost_per_mile: Average carrier cost per mile (default: $2.00)
            default_profit_margin: Target profit margin as decimal (default: 0.20 = 20%)
        """
        self.logger = logging.getLogger(__name__)

        if cost_per_mile <= 0:
            raise ValueError("Cost per mile must be positive")
        if default_profit_margin < 0:
            raise ValueError("Profit margin cannot be negative")
        self.cost_per_mile = cost_per_mile
        self.default_profit_margin = default_profit_margin
        self.max_profit_margin = max_profit_margin

    async def calculate_bid(self,
                            loaded_miles: int,
                            location_risk: LocationRisk = LocationRisk.HIGH_RISK,
                            profit_margin: Optional[float] = None,
                            round_to: int = 10) -> BidResult:
        """
        Calculate a bid based on loaded miles and location risk.

        Args:
            loaded_miles: Number of loaded (revenue) miles
            location_risk: Risk level of pickup location (default: STANDARD)
            profit_margin: Custom profit margin, uses default if None
            round_to: Round final bid to nearest value (default: 10)

        Returns:
            BidResult object containing all calculation details
        """
        if loaded_miles <= 0:
            raise ValueError("Loaded miles must be positive")

        profit = profit_margin if profit_margin is not None else self.default_profit_margin
        if profit < 0:
            raise ValueError("Profit margin cannot be negative")

        # Step 1: Calculate risk-adjusted miles
        multiplier = location_risk.multiplier
        est_total_miles = loaded_miles * multiplier

        # Step 2: Calculate break-even cost
        break_even = est_total_miles * self.cost_per_mile

        # Step 3: Add target profit
        calculated_bid = break_even * (1 + profit)
        max_bid = break_even * (1 + self.max_profit_margin)

        # Step 4: Round final bid
        min_rate = self._round_up(calculated_bid, round_to)
        max_rate = self._round_up(max_bid, round_to)

        # Calculate actual profit metrics
        # expected_profit = final_bid - break_even
        # profit_margin_percent = (expected_profit / break_even) * 100
        print(f"Calculated bid for {loaded_miles} miles at {location_risk.description}: ${min_rate:.2f} (Max: ${max_rate:.2f}) break_even_cost=${break_even:.2f} target_profit_margin={profit:.0f}% safety_multiplier={multiplier}x estimated_total_miles={est_total_miles:.0f} cost_per_mile=${self.cost_per_mile:.2f} calculated_bid=${calculated_bid:.2f}")
        return BidResult(
            loaded_miles=loaded_miles,
            safety_multiplier=multiplier,
            estimated_total_miles=est_total_miles,
            cost_per_mile=self.cost_per_mile,
            break_even_cost=break_even,
            target_profit_margin=profit,
            calculated_bid=calculated_bid,
            min_rate=min_rate,
            max_rate=max_rate,
            # expected_profit=expected_profit,
            # profit_margin_percent=profit_margin_percent
        )

    @staticmethod
    def _round_up(value: float, round_to: int) -> float:
        """Round up to the nearest multiple of round_to"""
        import math
        return math.ceil(value / round_to) * round_to

