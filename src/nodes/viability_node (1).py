"""Viability evaluation node - Refined with deep financial modeling

Save this file as: src/nodes/viability_node.py
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


class ViabilityAnalyzer:
    """Advanced viability analyzer with financial modeling"""

    def __init__(self):
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.GROQ_MODEL,
            temperature=0.7
        )
        self.search_tool = SearchTool()

    def research_market_size(self, startup_idea: str, customer_segment: str) -> Dict:
        """Phase 1: Research market size and customer count"""
        logger.info("Phase 1: Researching market size...")

        search_query = f"{customer_segment} market size TAM customers 2024"
        market_data = self.search_tool.search(search_query, topic="general")

        prompt = ChatPromptTemplate.from_template("""
        Research the market size and estimate customer count.
        
        Startup Idea: {startup_idea}
        Customer Segment: {customer_segment}
        Market Research: {market_data}
        
        Provide:
        1. Total Addressable Market (TAM)
        2. Serviceable Addressable Market (SAM)
        3. Estimated Total Customer Count (TAM and SAM)
        4. Customer Distribution
        5. Market Growth Rate
        6. Market Maturity
        
        Format as JSON with keys: tam, sam, total_customers_tam, total_customers_sam, distribution, growth_rate, maturity
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea,
            "customer_segment": customer_segment,
            "market_data": str(market_data[:3]) if market_data else "No market data"
        })

        market_size_data = extract_json(
            response.content,
            fallback_keys=("tam", "sam", "total_customers_tam", "total_customers_sam", "distribution", "growth_rate", "maturity"),
        )

        logger.info(f"Market size researched")
        return market_size_data

    def determine_funding_strategy(self, startup_idea: str, funding_model: str = None, arr_target_input: str = None) -> Dict:
        """Phase 2: Funding strategy and ARR goals.

        No longer blocks on input(). `funding_model` and `arr_target_input`
        are supplied by the caller (the CLI asks for them up front and passes
        them in via state; the API accepts them as optional fields on the
        /evaluate/start request body). Falls back to sensible defaults --
        bootstrap / $1M ARR -- when not provided, and logs (instead of
        interactively confirming) when a target looks mismatched with the
        chosen funding model, so nothing here can block waiting on stdin.
        """
        logger.info("Phase 2: Determining funding strategy...")

        funding_model = (funding_model or "bootstrap").strip().lower()
        if funding_model not in ("bootstrap", "vc"):
            logger.warning(f"Unrecognized funding_model '{funding_model}', defaulting to 'bootstrap'")
            funding_model = "bootstrap"

        arr_targets = {"100k": 100000, "1m": 1000000, "10m": 10000000}
        arr_input = (arr_target_input or "1m").strip().lower().replace("$", "").replace(",", "")

        if arr_input in arr_targets:
            arr_target = arr_targets[arr_input]
            arr_reasoning = f"Standard milestone: {arr_input} ARR"
        else:
            try:
                arr_target = float(arr_input)
                arr_reasoning = f"Custom target: ${arr_target:,.0f}"
            except ValueError:
                arr_target = 1000000
                arr_reasoning = "Default: $1M (unparseable ARR target)"

        if funding_model == "bootstrap" and arr_target > 5000000:
            logger.warning(f"${arr_target:,.0f} ARR target is ambitious for a bootstrapped strategy")

        if funding_model == "vc" and arr_target < 1000000:
            logger.warning(f"${arr_target:,.0f} ARR target may not justify VC funding")

        strategy_data = {
            "funding_model": funding_model,
            "arr_target": arr_target,
            "arr_reasoning": arr_reasoning,
            "timeframe_years": 3
        }

        logger.info(f"Funding strategy: {funding_model}, ARR: ${arr_target:,.0f}")
        return strategy_data

    def estimate_pricing_and_customers(self, startup_idea: str, strategy_data: Dict) -> Dict:
        """Phase 3: Pricing and customer metrics. Formula: Active Customers = ARR / Yearly ACR"""
        logger.info("Phase 3: Estimating pricing and customers...")

        prompt = ChatPromptTemplate.from_template("""
        Estimate pricing and calculate required active customers.
        
        Startup Idea: {startup_idea}
        ARR Target: ${arr_target:,.0f}
        Funding: {funding_model}
        
        Provide:
        1. Pricing Strategy (per-seat, freemium, usage-based, etc.)
        2. Monthly/Yearly price estimation
        3. Required active customers using: Active Customers = {arr_target:,.0f} / Yearly Customer Revenue
        4. 3 scenarios: conservative, realistic, optimistic
        
        Format as JSON with keys: pricing_model, monthly_price, yearly_acr, scenarios
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea,
            "arr_target": strategy_data["arr_target"],
            "funding_model": strategy_data["funding_model"]
        })

        pricing_data = extract_json(
            response.content,
            fallback_keys=("pricing_model", "monthly_price", "yearly_acr", "scenarios"),
        )

        logger.info("Pricing estimated")
        return pricing_data

    def research_churn_rate(self, customer_segment: str, industry: str) -> Dict:
        """Phase 4: Churn research. Formula: Churn Rate = 1 / Customer Lifetime (months)"""
        logger.info("Phase 4: Researching churn rates...")

        search_query = f"{industry} SaaS customer churn retention benchmark"
        churn_data = self.search_tool.search(search_query, topic="general")

        prompt = ChatPromptTemplate.from_template("""
        Research industry churn benchmarks.
        
        Customer Segment: {customer_segment}
        Industry: {industry}
        
        Provide:
        1. Benchmark Churn Rate
        2. Customer Lifetime (months)
        3. Formula: Churn = 1 / Lifetime (months)
        4. Factors affecting churn
        5. Improvement opportunities
        6. 3 scenarios: high churn, realistic, low churn
        
        Format as JSON with keys: benchmark_churn, customer_lifetime_months, scenarios
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "customer_segment": customer_segment,
            "industry": industry
        })

        churn_research = extract_json(
            response.content,
            fallback_keys=("benchmark_churn", "customer_lifetime_months", "scenarios"),
        )

        logger.info("Churn research completed")
        return churn_research

    def calculate_customer_acquisition_funnel(self, startup_idea: str, pricing_data: Dict, churn_data: Dict) -> Dict:
        """Phase 5: Customer acquisition funnel - The Customer Factory"""
        logger.info("Phase 5: Calculating acquisition funnel...")

        prompt = ChatPromptTemplate.from_template("""
        Calculate the customer acquisition funnel.
        
        Startup Idea: {startup_idea}
        
        Design:
        1. Acquisition Rate: User acquisition
        2. Activation Rate: Trial/pilot conversion (10-30%)
        3. Revenue Rate: Trial to paying (5-15%)
        4. Overall Conversion: Leads to Customers (~1%)
        
        Calculate:
        - Monthly leads needed
        - CAC (Customer Acquisition Cost)
        - LTV (Lifetime Value)
        - Payback Period
        - Referral impact
        
        Provide 3 scenarios: conservative, realistic, aggressive
        
        Format as JSON with keys: acquisition_rate, activation_rate, revenue_rate, scenarios, referral_impact
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea
        })

        funnel_data = extract_json(
            response.content,
            fallback_keys=("acquisition_rate", "activation_rate", "revenue_rate", "scenarios", "referral_impact"),
        )

        logger.info("Acquisition funnel calculated")
        return funnel_data

    def generate_viability_assessment(self, market_data: Dict, strategy_data: Dict, pricing_data: Dict,
                                     churn_data: Dict, funnel_data: Dict) -> Dict:
        """Phase 6: Generate viability assessment"""
        logger.info("Phase 6: Generating viability assessment...")

        prompt = ChatPromptTemplate.from_template("""
        Generate viability assessment (0-100).
        
        Market Data: {market_data}
        Funding Strategy: {strategy_data}
        Pricing Data: {pricing_data}
        Churn Data: {churn_data}
        Acquisition Funnel: {funnel_data}
        
        Assess:
        1. Business Model Soundness (25 pts): Pricing, CAC/LTV, payback
        2. Market Opportunity (25 pts): TAM sufficiency, achievable customers, growth
        3. Unit Economics (25 pts): Revenue per customer, churn, margins
        4. Go-to-Market (25 pts): Funnel achievable, referrals viable, realistic CAC
        
        Provide:
        - Viability Score (0-100)
        - Score breakdown
        - Financial metrics summary
        - Strengths and risks
        - Assumptions to validate
        - Funding model fit
        
        Format as JSON with keys: overall_score, breakdown, financial_summary, strengths, risks, assumptions, funding_fit
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "market_data": json.dumps(market_data),
            "strategy_data": json.dumps(strategy_data),
            "pricing_data": json.dumps(pricing_data),
            "churn_data": json.dumps(churn_data),
            "funnel_data": json.dumps(funnel_data)
        })

        viability_assessment = extract_json(
            response.content,
            fallback_keys=("overall_score", "breakdown", "financial_summary", "strengths", "risks", "assumptions", "funding_fit"),
        )

        logger.info(f"Viability Score: {viability_assessment.get('overall_score', 0)}/100")
        return viability_assessment


def viability_node(state: StartupStressTestState) -> StartupStressTestState:
    """Comprehensive viability evaluation node."""
    logger.info(f"Starting viability evaluation for: {state['startup_idea']}")

    analyzer = ViabilityAnalyzer()

    customer_segment = "target market"
    industry = state['startup_idea']

    market_data = analyzer.research_market_size(state['startup_idea'], customer_segment)
    strategy_data = analyzer.determine_funding_strategy(
        state['startup_idea'],
        funding_model=state.get('funding_model'),
        arr_target_input=state.get('arr_target'),
    )
    pricing_data = analyzer.estimate_pricing_and_customers(state['startup_idea'], strategy_data)
    churn_data = analyzer.research_churn_rate(customer_segment, industry)
    funnel_data = analyzer.calculate_customer_acquisition_funnel(state['startup_idea'], pricing_data, churn_data)
    viability_assessment = analyzer.generate_viability_assessment(market_data, strategy_data, pricing_data, churn_data, funnel_data)

    analysis_report = f"""
    ============================================
    COMPREHENSIVE VIABILITY ANALYSIS
    ============================================
    
    PHASE 1: MARKET SIZE RESEARCH
    {json.dumps(market_data, indent=2)}
    
    PHASE 2: FUNDING STRATEGY & ARR GOALS
    {json.dumps(strategy_data, indent=2)}
    
    PHASE 3: PRICING & CUSTOMER METRICS
    {json.dumps(pricing_data, indent=2)}
    
    PHASE 4: CHURN RATE & CUSTOMER LIFETIME
    {json.dumps(churn_data, indent=2)}
    
    PHASE 5: CUSTOMER ACQUISITION FUNNEL
    {json.dumps(funnel_data, indent=2)}
    
    PHASE 6: VIABILITY ASSESSMENT
    {json.dumps(viability_assessment, indent=2)}
    ============================================
    """

    state['viability_analysis'] = analysis_report
    state['viability_status'] = EvaluationStatus.COMPLETED
    state['viability_score'] = viability_assessment.get('overall_score', 0)

    logger.info(f"Viability evaluation completed. Score: {state['viability_score']}")
    return state
