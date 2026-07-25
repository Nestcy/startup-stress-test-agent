"""Desirability evaluation node - Refined with deeper customer and market research

Save this file as: src/nodes/desirability_node.py
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


class DesirabilityAnalyzer:
    """Advanced desirability analyzer with multi-phase research"""

    def __init__(self):
        self.llm = ChatGroq(
            api_key=Config.GROQ_API_KEY,
            model=Config.GROQ_MODEL,
            temperature=0.7
        )
        self.search_tool = SearchTool()

    def identify_customer_segment(self, startup_idea: str, idea_description: str) -> Dict:
        """Phase 1: Identify who the primary customer is"""
        logger.info("Phase 1: Identifying customer segment...")

        prompt = ChatPromptTemplate.from_template("""
        Based on this startup idea, identify the primary customer segment.
        
        Startup Idea: {startup_idea}
        Description: {description}
        
        Analyze and provide:
        1. Primary Customer Segment: Who are the main users?
        2. Customer Profile: Demographics, psychographics, behaviors
        3. Customer Size: Estimated TAM (Total Addressable Market)
        4. Customer Pain Level: How acute is their pain?
        5. Customer Willingness to Pay: Budget/spending capacity
        
        Format as JSON with keys: segment, profile, tam, pain_level, willingness_to_pay
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea,
            "description": idea_description
        })

        customer_data = extract_json(
            response.content,
            fallback_keys=("segment", "profile", "tam", "pain_level", "willingness_to_pay"),
        )

        logger.info(f"Customer segment identified: {customer_data.get('segment', 'Unknown')}")
        return customer_data

    def research_market_shifts(self, startup_idea: str, customer_segment: str) -> Dict:
        """Phase 2: What has changed in the world + switching triggers"""
        logger.info("Phase 2: Researching market shifts and switching triggers...")

        search_query = f"{startup_idea} market trends industry shifts changes 2024 2025"
        market_trends = self.search_tool.search(search_query, topic="general")

        prompt = ChatPromptTemplate.from_template("""
        Analyze what has changed in the world that makes this startup idea timely.
        
        Startup Idea: {startup_idea}
        Target Customer: {customer_segment}
        Recent Market Data: {market_data}
        
        Identify:
        1. Major Market Shifts: What has fundamentally changed?
        2. Primary Switching Triggers: What makes customers switch NOW?
        3. Secondary Switching Triggers: What else could accelerate adoption?
        4. Urgency Timeline: How urgent is this shift?
        
        Format as JSON with keys: major_shifts, primary_trigger, secondary_triggers, urgency_timeline
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea,
            "customer_segment": customer_segment,
            "market_data": str(market_trends[:3]) if market_trends else "No market data"
        })

        shifts_data = extract_json(
            response.content,
            fallback_keys=("major_shifts", "primary_trigger", "secondary_triggers", "urgency_timeline"),
        )

        logger.info("Market shifts and switching triggers identified")
        return shifts_data

    def research_existing_alternatives(self, startup_idea: str, customer_segment: str) -> Dict:
        """Phase 3: What existing alternatives do customers use?"""
        logger.info("Phase 3: Researching existing alternatives...")

        search_query = f"how {customer_segment} currently solve {startup_idea} alternatives tools"
        alt_results = self.search_tool.search(search_query, topic="general")

        prompt = ChatPromptTemplate.from_template("""
        Identify existing alternatives and workarounds customers currently use.
        
        Startup Idea: {startup_idea}
        Customer Segment: {customer_segment}
        Market Research: {research_data}
        
        Research and document:
        1. Direct Competitors: Existing products solving this exactly
        2. Indirect Competitors: Adjacent solutions or workarounds
        3. Status Quo: What % still haven't adopted any solution?
        4. Customer Satisfaction with Alternatives
        5. Switching Costs
        
        Format as JSON with keys: direct_competitors, indirect_competitors, status_quo, satisfaction, switching_costs
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea,
            "customer_segment": customer_segment,
            "research_data": str(alt_results[:3]) if alt_results else "No research data"
        })

        alternatives_data = extract_json(
            response.content,
            fallback_keys=("direct_competitors", "indirect_competitors", "status_quo", "satisfaction", "switching_costs"),
        )

        logger.info("Existing alternatives documented")
        return alternatives_data

    def analyze_solution_fit(self, startup_idea: str, idea_description: str,
                            customer_segment: str, alternatives_data: Dict) -> Dict:
        """Phase 4: Founder's identified problems and proposed solution"""
        logger.info("Phase 4: Analyzing solution-problem fit...")

        prompt = ChatPromptTemplate.from_template("""
        Analyze the startup's solution against identified customer problems.
        
        Startup Idea: {startup_idea}
        Solution Description: {description}
        Target Customer: {customer_segment}
        
        Evaluate:
        1. Problem Validation: Are these real problems?
        2. Solution Uniqueness: How does this solution differ?
        3. Solution-Problem Fit: Does solution actually solve problems?
        4. Adoption Advantage: Why would customers switch?
        5. Desirability Risk Assessment
        
        Format as JSON with keys: problem_validation, uniqueness, solution_fit, adoption_advantage, risk_assessment
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "startup_idea": startup_idea,
            "description": idea_description,
            "customer_segment": customer_segment
        })

        solution_fit_data = extract_json(
            response.content,
            fallback_keys=("problem_validation", "uniqueness", "solution_fit", "adoption_advantage", "risk_assessment"),
        )

        logger.info("Solution-problem fit analyzed")
        return solution_fit_data

    def generate_desirability_score(self, customer_data: Dict, shifts_data: Dict,
                                    alternatives_data: Dict, solution_fit_data: Dict) -> Dict:
        """Phase 5: Generate comprehensive desirability score and analysis"""
        logger.info("Phase 5: Generating desirability analysis...")

        prompt = ChatPromptTemplate.from_template("""
        Based on comprehensive research, calculate desirability score (0-100).
        
        Customer Research: {customer_data}
        Market Shifts: {shifts_data}
        Existing Alternatives: {alternatives_data}
        Solution Fit: {solution_fit_data}
        
        Evaluate by:
        1. Customer Demand (25 points): TAM, pain level, willingness to pay
        2. Market Timing (25 points): Shifts and switching triggers
        3. Competitive Positioning (25 points): Satisfaction gaps, uniqueness
        4. Solution-Problem Fit (25 points): Coverage and adoption advantage
        
        Provide:
        - Overall Desirability Score (0-100)
        - Score breakdown by category
        - Key strengths and risks
        - Critical assumptions to validate
        - Recommended validation experiments
        
        Format as JSON with keys: overall_score, breakdown, strengths, risks, assumptions, validation_experiments
        """)

        chain = prompt | self.llm
        response = chain.invoke({
            "customer_data": json.dumps(customer_data),
            "shifts_data": json.dumps(shifts_data),
            "alternatives_data": json.dumps(alternatives_data),
            "solution_fit_data": json.dumps(solution_fit_data)
        })

        final_score_data = extract_json(
            response.content,
            fallback_keys=("overall_score", "breakdown", "strengths", "risks", "assumptions", "validation_experiments"),
        )

        logger.info(f"Desirability Score: {final_score_data.get('overall_score', 0)}/100")
        return final_score_data


def desirability_node(state: StartupStressTestState) -> StartupStressTestState:
    """Comprehensive desirability evaluation node."""
    logger.info(f"Starting desirability evaluation for: {state['startup_idea']}")

    analyzer = DesirabilityAnalyzer()

    customer_data = analyzer.identify_customer_segment(state['startup_idea'], state.get('idea_description', ''))
    customer_segment = customer_data.get('segment', 'Unknown segment')
    shifts_data = analyzer.research_market_shifts(state['startup_idea'], customer_segment)
    alternatives_data = analyzer.research_existing_alternatives(state['startup_idea'], customer_segment)
    solution_fit_data = analyzer.analyze_solution_fit(state['startup_idea'], state.get('idea_description', ''), customer_segment, alternatives_data)
    final_analysis = analyzer.generate_desirability_score(customer_data, shifts_data, alternatives_data, solution_fit_data)

    analysis_report = f"""
    ============================================
    COMPREHENSIVE DESIRABILITY ANALYSIS
    ============================================
    
    PHASE 1: CUSTOMER IDENTIFICATION
    {json.dumps(customer_data, indent=2)}
    
    PHASE 2: MARKET SHIFTS & SWITCHING TRIGGERS
    {json.dumps(shifts_data, indent=2)}
    
    PHASE 3: EXISTING ALTERNATIVES ANALYSIS
    {json.dumps(alternatives_data, indent=2)}
    
    PHASE 4: SOLUTION-PROBLEM FIT
    {json.dumps(solution_fit_data, indent=2)}
    
    PHASE 5: DESIRABILITY SCORE & RECOMMENDATIONS
    {json.dumps(final_analysis, indent=2)}
    ============================================
    """

    state['desirability_analysis'] = analysis_report
    state['desirability_status'] = EvaluationStatus.COMPLETED
    state['desirability_score'] = final_analysis.get('overall_score', 0)

    logger.info(f"Desirability evaluation completed. Score: {state['desirability_score']}")
    return state
