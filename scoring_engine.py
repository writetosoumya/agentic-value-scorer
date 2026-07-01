"""
scoring_engine.py
ARM™ and AVRE™ scoring logic — updated to match published framework definitions.

ARM™ domains (published May 26, 2026 — Issue 2):
  execution_integrity, decision_integrity, capability_debt,
  knowledge_sustainability, enterprise_adaptation

AVRE™ lenses (published June 25, 2026 — Issue 4):
  ROI (financial), ROE (operational), ROF (strategic/future)
  NEV = (ROI + ROE + ROF) − Total Cost of Agentification − ARM™ Risk Penalty
"""

from dataclasses import dataclass, field
from typing import Dict, List
import json


@dataclass
class ARMScores:
    execution_integrity:     float = 0.0   # Can work still be executed correctly?
    decision_integrity:      float = 0.0   # Are humans deciding or merely approving?
    capability_debt:         float = 0.0   # What capabilities are being consumed?
    knowledge_sustainability: float = 0.0  # Is knowledge preserved or locked in agents?
    enterprise_adaptation:   float = 0.0   # Can org still respond when reality changes?

    def composite(self) -> float:
        # Published ARM™ domain weights
        return round(
            self.execution_integrity      * 0.25
            + self.decision_integrity     * 0.25
            + self.capability_debt        * 0.20
            + self.knowledge_sustainability * 0.15
            + self.enterprise_adaptation  * 0.15,
            2,
        )

    def risk_tier(self) -> str:
        s = self.composite()
        if s <= 3.0:  return "🟢 GREEN — Proceed with standard governance"
        elif s <= 5.5: return "🟡 AMBER — Proceed with enhanced controls"
        elif s <= 7.5: return "🔴 RED — Phased deployment only. Independent review required."
        else:          return "⛔ CRITICAL — Do not deploy without executive sign-off and external audit"

    def risk_multiplier(self) -> float:
        """ARM™ Risk Multiplier applied as penalty in AVRE™ NEV calculation."""
        return round(1 + (self.composite() / 10), 3)


@dataclass
class AVREScores:
    roi_score:       float = 0.0   # Return on Investment — financial lens
    roe_score:       float = 0.0   # Return on Effort — operational lens
    rof_score:       float = 0.0   # Return on Future — strategic lens
    opportunity_cost: float = 0.0  # Urgency of acting now (reference, not in NEV)

    def benefit_realization(self) -> float:
        """Weighted combination of three AVRE™ lenses."""
        return round(
            self.roi_score * 0.40
            + self.roe_score * 0.30
            + self.rof_score * 0.30,
            2,
        )


@dataclass
class ScoringResult:
    use_case_name:        str
    use_case_description: str
    arm:  ARMScores = field(default_factory=ARMScores)
    avre: AVREScores = field(default_factory=AVREScores)
    rationale:       Dict[str, str] = field(default_factory=dict)
    rag_sources:     List[str]      = field(default_factory=list)
    recommendations: List[str]      = field(default_factory=list)

    def net_enterprise_value(self) -> float:
        """
        NEV = (ROI + ROE + ROF) − Total Cost of Agentification − ARM™ Risk Penalty
        Normalised to 0–10 for the AVS UI. Maps to AVRE™ 0–100 scale × 0.1.
        """
        capability_debt = self.arm.capability_debt
        multiplier      = self.arm.risk_multiplier()
        br              = self.avre.benefit_realization()
        raw_nev         = br - (capability_debt * multiplier * 0.15)
        return round(max(0.0, min(10.0, raw_nev * 1.2)), 2)

    def nev_rating(self) -> str:
        nev = self.net_enterprise_value()
        # Mapped from published AVRE™ 0–100 scale to 0–10
        if nev >= 9.0:  return "⭐ TRANSFORMATIONAL — Scale it"
        elif nev >= 7.5: return "✅ HIGH VALUE — Grow with confidence"
        elif nev >= 6.0: return "📈 VALUABLE — Optimize first"
        elif nev >= 4.0: return "⚠️  MARGINAL — Redesign before you expand"
        else:            return "❌ POOR — Retire it"

    def to_dict(self) -> dict:
        return {
            "use_case":    self.use_case_name,
            "description": self.use_case_description,
            "arm_scores": {
                "execution_integrity":      self.arm.execution_integrity,
                "decision_integrity":       self.arm.decision_integrity,
                "capability_debt":          self.arm.capability_debt,
                "knowledge_sustainability": self.arm.knowledge_sustainability,
                "enterprise_adaptation":    self.arm.enterprise_adaptation,
                "composite_arm_score":      self.arm.composite(),
                "risk_tier":                self.arm.risk_tier(),
                "risk_multiplier":          self.arm.risk_multiplier(),
            },
            "avre_scores": {
                "roi_score":           self.avre.roi_score,
                "roe_score":           self.avre.roe_score,
                "rof_score":           self.avre.rof_score,
                "opportunity_cost":    self.avre.opportunity_cost,
                "benefit_realization": self.avre.benefit_realization(),
            },
            "net_enterprise_value": self.net_enterprise_value(),
            "nev_rating":           self.nev_rating(),
            "rationale":            self.rationale,
            "recommendations":      self.recommendations,
            "rag_sources_used":     self.rag_sources,
        }

    def pretty_print(self):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich import box

        console = Console()
        console.print(f"\n[bold cyan]═══ AGENTIFICATION VALUE SCORE REPORT ═══[/bold cyan]")
        console.print(f"[bold]Use Case:[/bold] {self.use_case_name}")
        console.print(f"[dim]{self.use_case_description}[/dim]\n")

        arm_table = Table(title="ARM™ Risk Assessment", box=box.ROUNDED)
        arm_table.add_column("Domain", style="bold")
        arm_table.add_column("Score", justify="center")
        arm_table.add_column("Weight", justify="center")
        arm_table.add_column("Rationale", max_width=55)

        def sc(s):
            if s <= 3: return "green"
            elif s <= 5.5: return "yellow"
            elif s <= 7.5: return "red"
            else: return "bright_red"

        dims = [
            ("Execution Integrity",      self.arm.execution_integrity,      "25%", "execution_integrity"),
            ("Decision Integrity",        self.arm.decision_integrity,       "25%", "decision_integrity"),
            ("Capability Debt",           self.arm.capability_debt,          "20%", "capability_debt"),
            ("Knowledge Sustainability",  self.arm.knowledge_sustainability, "15%", "knowledge_sustainability"),
            ("Enterprise Adaptation",     self.arm.enterprise_adaptation,    "15%", "enterprise_adaptation"),
        ]
        for label, score, wt, key in dims:
            arm_table.add_row(label, f"[{sc(score)}]{score:.1f}[/{sc(score)}]", wt, self.rationale.get(key, "—")[:80])
        arm_table.add_section()
        arm_table.add_row("[bold]COMPOSITE ARM™ SCORE[/bold]", f"[bold]{self.arm.composite():.2f}[/bold]", "100%", self.arm.risk_tier())
        console.print(arm_table)

        avre_table = Table(title="AVRE™ Value Assessment", box=box.ROUNDED)
        avre_table.add_column("Lens", style="bold")
        avre_table.add_column("Score", justify="center")
        avre_table.add_column("Weight", justify="center")
        avre_table.add_column("Rationale", max_width=55)

        lenses = [
            ("ROI — Return on Investment",      self.avre.roi_score,        "40%", "roi"),
            ("ROE — Return on Effort",           self.avre.roe_score,        "30%", "roe"),
            ("ROF — Return on Future",           self.avre.rof_score,        "30%", "rof"),
            ("Opportunity Cost (urgency ref)",   self.avre.opportunity_cost, "ref", "opportunity_cost"),
        ]
        def vc(s):
            if s >= 7: return "green"
            elif s >= 5: return "cyan"
            elif s >= 3: return "yellow"
            return "red"
        for label, score, wt, key in lenses:
            avre_table.add_row(label, f"[{vc(score)}]{score:.1f}[/{vc(score)}]", wt, self.rationale.get(key, "—")[:80])
        avre_table.add_section()
        avre_table.add_row("[bold]BENEFIT REALIZATION[/bold]", f"[bold]{self.avre.benefit_realization():.2f}[/bold]", "", "")
        console.print(avre_table)

        nev = self.net_enterprise_value()
        nc  = "green" if nev >= 7.5 else "cyan" if nev >= 6 else "yellow" if nev >= 4 else "red"
        console.print(Panel(
            f"[bold {nc}]NET ENTERPRISE VALUE: {nev:.2f} / 10[/bold {nc}]\n"
            f"{self.nev_rating()}\n\n"
            f"ARM™ Risk Multiplier: {self.arm.risk_multiplier():.3f}  |  "
            f"Benefit Realization: {self.avre.benefit_realization():.2f}\n"
            f"Sources: {', '.join(self.rag_sources)}",
            title="[bold]AVRE™ Final Verdict[/bold]",
            border_style=nc,
        ))
        if self.recommendations:
            console.print("\n[bold]📋 Recommendations:[/bold]")
            for i, rec in enumerate(self.recommendations, 1):
                console.print(f"  {i}. {rec}")
        console.print()
