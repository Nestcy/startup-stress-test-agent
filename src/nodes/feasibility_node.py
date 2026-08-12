"""Feasibility evaluation node - Refined with traction roadmap and product-market fit milestones

Save this file as: src/nodes/feasibility_node.py
"""
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.state import StartupStressTestState, EvaluationStatus
from src.tools.search_tool import SearchTool
from src.utils.logger import logger
from src.utils.config import Config
from src.utils.llm_json import extract_json
from typing import Dict
import json


class FeasibilityAnalyzer:
    """Advanced feasibility analyzer with traction roadmap and PMF milestones"""

    def __init__(self):
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.GROQ_MODEL,
            temperature=0.7
        )
        self.search_tool = SearchTool()

    def research_technical_requirements(self, startup_idea: str, desirability_analysis: str) -> Dict:
        """Phase 1: Research technical requirements and architecture"""
        logger.info("Phase 1: Researching technical requirements...")

        search_query = f"how to build {startup_idea} technology stack"
        tech_data = self.search_tool.search(search_query, topic="general")

        prompt = ChatPromptTemplate.from_template("""
        Analyze technical requirements to build this startup.
        
        Startup Idea: {startup_idea}
        
        Provide:
        1. Core Technical Components
        2. Recommended Tech Stack (frontend, backend, database, infrastructure)
        3. MVP Scope (what can be cut to ship faster)
        4. Technical Risks and scalability
        5. Team Requirements and hiring timeline
        
        Format as JSON with keys: core_components, tech_stack, mvp_scope, technical_risks,
        team_requirements, sourced_claims, assumptions
        "sourced_claims": specific facts above drawn directly from the tech research provided.
        "assumptions": specific facts above that are your own estimate (e.g. team hiring
        timeline), not confirmed by the research.
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea
        })

        tech_requirements = extract_json(
            response.content,
            fallback_keys=("core_components", "tech_stack", "mvp_scope", "technical_risks", "team_requirements", "sourced_claims", "assumptions"),
        )

        logger.info("Technical requirements researched")
        return tech_requirements

    def build_traction_roadmap(self, startup_idea: str, viability_analysis: str) -> Dict:
        """Phase 2: Build traction roadmap with 10x growth milestones at 3, 12, 24, 36 months"""
        logger.info("Phase 2: Building 10x growth traction roadmap...")

        prompt = ChatPromptTemplate.from_template("""
        Build 36-month traction roadmap with 10x growth at each milestone.
        
        Startup Idea: {startup_idea}
        
        Milestones:
        
        MONTH 3 (Problem/Solution Fit):
        - Customers: 5-10 early adopters
        - Revenue: $500-2K MRR (validate demand through revenue)
        - Metrics: NPS 7+, problem validated, churn monitoring
        
        MONTH 12 (Product/Market Fit Launch) - 10x from Month 3:
        - Customers: 50-100 (10x)
        - Revenue: $10K+ MRR (10x)
        - Metrics: MRR growth 20-30%, churn <5%, organic >20%, NPS 8+
        
        MONTH 24 (Product/Market Fit Refinement) - 10x from Month 12:
        - Customers: 500-1000 (10x)
        - Revenue: $100K+ MRR (10x)
        - Metrics: MRR growth 15-25%, churn <3%, LTV:CAC >3:1
        
        MONTH 36 (Scale & Growth Rockets) - 10x from Month 24:
        - Customers: 5K-10K (10x)
        - Revenue: $1M+ MRR (10x)
        - Metrics: MRR growth 10-20%, churn <2%, LTV:CAC >4:1, 3+ growth channels
        
        For each milestone provide: targets, key activities, resources, risks
        
        Format as JSON with keys: milestone_3, milestone_12, milestone_24, milestone_36
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea
        })

        roadmap = extract_json(
            response.content,
            fallback_keys=("milestone_3", "milestone_12", "milestone_24", "milestone_36"),
        )

        logger.info("Traction roadmap with 10x milestones created")
        return roadmap

    def create_now_next_later_plan(self, startup_idea: str) -> Dict:
        """Phase 3: Create NOW, NEXT, LATER product rollout plan"""
        logger.info("Phase 3: Creating NOW, NEXT, LATER plan...")

        prompt = ChatPromptTemplate.from_template("""
        Create detailed NOW, NEXT, LATER rollout plan.
        
        Startup Idea: {startup_idea}
        
        ========== NOW (0-3 MONTHS: PROBLEM/SOLUTION FIT) ==========
        PRIMARY GOAL: Validate real demand through revenue
        
        Activities:
        1. Problem Discovery (Weeks 1-4): Interview 20-30 customers
        2. Solution Design (Weeks 5-8): Create mockup/prototype
        3. Demo-Sell (Weeks 9-12): Close first 5-10 paying customers
        
        Success Metrics:
        - Paying customers: 5-10
        - First MRR: $500-2K
        - Revenue validation: YES
        
        ========== NEXT (3-24 MONTHS: PRODUCT/MARKET FIT) ==========
        PRIMARY GOAL: Build product customers love
        
        Phase 1 (Months 3-12) - Build & Launch:
        - Build MVP, launch to early adopters
        - Weekly refinements based on usage
        - Target: 50-100 paying customers, $10K MRR
        
        Phase 2 (Months 12-24) - Refine & Scale:
        - Refine based on analytics
        - Achieve PMF: NPS 50+, churn <5%, organic >20%
        - Target: 200-500+ customers, $10K-50K MRR
        
        ========== LATER (24-36 MONTHS: SCALE & GROWTH ROCKETS) ==========
        PRIMARY GOAL: Execute growth rockets for 10x scaling
        
        Identify Growth Rockets:
        - Best acquisition channel by LTV
        - Organic referral opportunities
        - Partnership/marketing options
        
        Execute:
        - Launch growth marketing
        - Build referral loops
        - Scale to 5K-10K customers, $1M+ MRR
        
        ========== GO/NO-GO GATES ==========
        Gate NOW→NEXT: 5+ customers, $500+ MRR
        Gate NEXT→LATER: PMF signals (NPS 50+, churn <5%, organic >20%)
        
        Provide detailed monthly activities, metrics, gates, and resources for each phase.
        
        Format as JSON with keys: now_phase, next_phase, later_phase, gates, key_metrics
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea
        })

        rollout_plan = extract_json(
            response.content,
            fallback_keys=("now_phase", "next_phase", "later_phase", "gates", "key_metrics"),
        )

        logger.info("NOW, NEXT, LATER rollout plan created")
        return rollout_plan

    def generate_feasibility_assessment(self, tech_requirements: Dict, traction_roadmap: Dict, rollout_plan: Dict) -> Dict:
        """Phase 4: Generate comprehensive feasibility assessment"""
        logger.info("Phase 4: Generating feasibility assessment...")

        prompt = ChatPromptTemplate.from_template("""
        Generate feasibility assessment (0-100).
        
        Technical Requirements: {tech_requirements}
        Traction Roadmap: {traction_roadmap}
        Rollout Plan: {rollout_plan}
        
        SCORING PHILOSOPHY (read this before scoring): Score whether this idea is
        buildable and executable with a reasonable team and timeline, based on the
        technical research above. Do NOT penalize the idea for not having a team or
        codebase yet -- that's true of every pre-launch idea and isn't itself a
        feasibility flaw. Score low only where the technology itself is genuinely
        hard to build (e.g. requires research-grade AI, regulatory approval, physical
        infrastructure) or the plan is genuinely unrealistic -- not merely because
        nothing has been built yet.
        
        Assess:
        1. Technical Feasibility (25 pts): Can tech be built? Realistic timeline?
        2. Execution Feasibility (25 pts): Team can execute? Resources clear?
        3. Market Fit Achievability (25 pts): Can PMF be achieved? Rollout realistic?
        4. Go-to-Market Readiness (25 pts): Sales clear? Growth rockets identified?
        
        Provide:
        - Feasibility Score (0-100)
        - Timeline summary (NOW/NEXT/LATER)
        - Key milestones and gates
        - Risks and assumptions
        - First steps recommendation
        - Overall assessment
        
        Format as JSON with keys: overall_score, breakdown, timeline_summary, milestones, risks, assumptions, first_steps, overall_assessment
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "tech_requirements": json.dumps(tech_requirements),
            "traction_roadmap": json.dumps(traction_roadmap),
            "rollout_plan": json.dumps(rollout_plan)
        })

        assessment = extract_json(
            response.content,
            fallback_keys=("overall_score", "breakdown", "timeline_summary", "milestones", "risks", "assumptions", "first_steps", "overall_assessment"),
        )

        logger.info(f"Feasibility Score: {assessment.get('overall_score', 0)}/100")
        return assessment


from src.nodes.desirability_node import _collect_provenance


def feasibility_node(state: StartupStressTestState) -> StartupStressTestState:
    """Comprehensive feasibility evaluation node."""
    logger.info(f"Starting feasibility evaluation for: {state['startup_idea']}")

    analyzer = FeasibilityAnalyzer()

    tech_requirements = analyzer.research_technical_requirements(state['startup_idea'], state.get('desirability_analysis', ''))
    traction_roadmap = analyzer.build_traction_roadmap(state['startup_idea'], state.get('viability_analysis', ''))
    rollout_plan = analyzer.create_now_next_later_plan(state['startup_idea'])
    feasibility_assessment = analyzer.generate_feasibility_assessment(tech_requirements, traction_roadmap, rollout_plan)

    provenance = []
    provenance += _collect_provenance("feasibility", "technical_requirements", tech_requirements)
    # The traction roadmap and NOW/NEXT/LATER phases are built from generic
    # SaaS growth benchmarks baked into the prompt template (e.g. "MRR
    # growth 20-30%"), not searched or specific to this idea -- flag them
    # as assumptions outright rather than treating them as researched.
    provenance.append({
        "stage": "feasibility", "phase": "traction_roadmap", "type": "assumption",
        "claim": "10x growth milestones and metric targets are generic SaaS benchmarks, not specific to this idea or market.",
    })
    provenance.append({
        "stage": "feasibility", "phase": "rollout_plan", "type": "assumption",
        "claim": "NOW/NEXT/LATER timeline and gate thresholds are generic startup benchmarks, not specific to this idea or market.",
    })
    for claim in (feasibility_assessment.get("assumptions") or []):
        provenance.append({"stage": "feasibility", "phase": "scoring", "type": "assumption", "claim": claim})

    analysis_report = f"""
    ============================================
    COMPREHENSIVE FEASIBILITY ANALYSIS
    ============================================
    
    PHASE 1: TECHNICAL REQUIREMENTS
    {json.dumps(tech_requirements, indent=2)}
    
    PHASE 2: TRACTION ROADMAP (10x GROWTH MILESTONES)
    {json.dumps(traction_roadmap, indent=2)}
    
    PHASE 3: NOW, NEXT, LATER ROLLOUT PLAN
    {json.dumps(rollout_plan, indent=2)}
    
    PHASE 4: FEASIBILITY ASSESSMENT
    {json.dumps(feasibility_assessment, indent=2)}
    ============================================
    """

    state['feasibility_analysis'] = analysis_report
    state['feasibility_status'] = EvaluationStatus.COMPLETED
    state['feasibility_score'] = feasibility_assessment.get('overall_score', 0)
    existing_provenance = [a for a in (state.get('all_assumptions') or []) if a.get('stage') != 'feasibility']
    state['all_assumptions'] = existing_provenance + provenance

    logger.info(f"Feasibility evaluation completed. Score: {state['feasibility_score']}")
    return state
