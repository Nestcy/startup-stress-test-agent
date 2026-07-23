# Startup Stress Test AI Agent

An AI-powered agent that evaluates startup ideas through three critical lenses: **Desirability**, **Viability**, and **Feasibility**. The agent uses LangChain + Groq LLM with human-in-the-loop evaluation, comprehensive financial modeling, traction roadmaps, and product-market fit planning.

## 🎯 Overview

This agent stress tests startup ideas by deeply analyzing:

### 🎯 DESIRABILITY: Does anyone want this?
- **5-Phase Customer Research:**
  - Customer identification and segmentation
  - Market shifts and switching triggers analysis
  - Existing alternatives and competitive landscape
  - Problem-solution fit validation
  - Comprehensive desirability scoring

### 💰 VIABILITY: Can this work as a business?
- **6-Phase Financial Modeling:**
  - Market size research (TAM/SAM)
  - **Interactive funding strategy** (Bootstrap vs. VC-backed)
  - Pricing model and customer metrics estimation
  - Industry churn rate and customer lifetime research
  - **Customer acquisition funnel** (10x detailed breakdown)
  - Business model viability assessment

### ⚙️ FEASIBILITY: Can we build it?
- **4-Phase Technical & Product Planning:**
  - Technical requirements and architecture
  - **10x growth traction roadmap** with milestones (3, 12, 24, 36 months)
  - **NOW, NEXT, LATER product rollout** (goal-oriented)
  - Product-market fit milestone tracking

## 🚀 Key Features

✅ **Multi-Phase LLM Analysis** - Groq ChatGroq (Qwen 3.6 27B)  
✅ **Real-Time Market Research** - Tavily Search API integration  
✅ **Human-in-the-Loop** - Interactive checkpoints at each gate  
✅ **Interactive Founder Interviews** - Funding strategy, ARR targets, rollout planning  
✅ **Financial Modeling** - Complete formulas for CAC, LTV, churn, ARR  
✅ **Traction Roadmap** - 10x growth targets at each milestone  
✅ **Comprehensive Reports** - Detailed analysis with recommendations  
✅ **Conditional Routing** - Smart early-exit for non-viable ideas  

## 📊 The Three Gates

```
Desirability Gate (Score < 30?) → EXIT
        ↓
Viability Gate (Score < 30 or weak combo?) → EXIT
        ↓
Feasibility Gate (Score < 25 or weak combo?) → EXIT
        ↓
Final Report & Recommendation
```

## 📋 Project Structure

```
startup-stress-test-agent/
├── src/
│   ├── __init__.py
│   ├── state.py                 # State schema and types
│   ├── graph.py                 # LangGraph orchestration with conditional edges
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── desirability_node.py # 5-phase customer research
│   │   ├── viability_node.py    # 6-phase financial modeling
│   │   ├── feasibility_node.py  # 4-phase tech + product planning
│   │   ├── human_review_node.py # Checkpoint interactions
│   │   └── report_node.py       # Final report generation
│   ├── tools/
│   │   ├── __init__.py
│   │   └── search_tool.py       # Tavily search wrapper
│   └── utils/
│       ├── __init__.py
│       ├── config.py            # Configuration management
│       └── logger.py            # Logging setup
├── requirements.txt
├── .env.example
├── main.py                      # Entry point
└── README.md
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nestcy/startup-stress-test-agent.git
   cd startup-stress-test-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Add your API keys to .env
   ```

## 🔑 Environment Variables

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
GROQ_MODEL=qwen/qwen3.6-27b
```

## 🚀 Usage

Run the agent:

```bash
python main.py
```

You'll be prompted to:
1. Enter your startup idea name
2. Describe the startup in detail
3. Answer interactive questions at each gate (desirability, viability, feasibility)
4. Receive a comprehensive stress test report with recommendations

## 📈 Workflow Breakdown

### DESIRABILITY (Phase 1)
**5-Phase Analysis:**
1. Customer identification and segmentation
2. Market shifts and switching triggers (primary & secondary)
3. Existing alternatives and competitive landscape
4. Solution-problem fit validation
5. Desirability score (0-100)

**Human Checkpoint:** Review and provide feedback

### VIABILITY (Phase 2)
**6-Phase Analysis:**
1. Market size research (TAM/SAM/customers)
2. **Interactive funding strategy** (Bootstrap vs. VC) + ARR goal setting
3. Pricing model and customer metrics (Active Customers = ARR / Yearly ACR)
4. Churn rate and customer lifetime research (Churn = 1 / Lifetime)
5. **Customer acquisition funnel** (Leads → Acquisition → Activation → Revenue)
   - CAC, LTV, Payback Period calculations
   - Referral impact analysis
6. Viability score (0-100)

**Human Checkpoint:** Review and provide feedback

### FEASIBILITY (Phase 3)
**4-Phase Analysis:**
1. Technical requirements and architecture
2. **Traction roadmap with 10x growth:**
   - Month 3: 5-10 customers, $500-2K MRR (Problem/Solution Fit)
   - Month 12: 50-100 customers, $10K+ MRR (10x)
   - Month 24: 500-1K customers, $100K+ MRR (10x)
   - Month 36: 5K-10K customers, $1M+ MRR (10x)
3. **NOW, NEXT, LATER rollout plan:**
   - NOW (3 months): Problem/Solution Fit - validate demand through revenue
   - NEXT (6-24 months): Product/Market Fit - build & refine based on usage
   - LATER (24-36 months): Scale & Growth Rockets - execute growth channels
4. Feasibility score (0-100)

**Human Checkpoint:** Review and provide feedback

### FINAL REPORT
- Executive summary
- Stress test results table
- Key strengths and risks
- Critical success factors
- Recommended next steps
- GO/CONDITIONAL/NO-GO recommendation
- Overall score (0-100)

## 🧮 Key Financial Formulas

**Active Customers Required:**
```
Active Customers = Annual Revenue Target ÷ Yearly Customer Revenue (ACR)
```

**Monthly Churn Rate:**
```
Churn Rate = 1 ÷ Customer Lifetime (in months)
```

**Customer Acquisition Funnel:**
```
Leads → Acquisition Rate → Activation Rate → Revenue Rate → Customers
Overall Conversion ≈ 1% (first principles)
```

**Payback Period:**
```
CAC Payback Period = Monthly CAC ÷ Monthly Revenue per Customer
Typical target: < 12 months
```

**Lifetime Value to CAC Ratio:**
```
LTV:CAC = Lifetime Value ÷ Customer Acquisition Cost
Target: > 3:1 (healthy), > 4:1 (excellent)
```

## 🎯 Conditional Gates

The agent intelligently exits early if:
- **Desirability < 30**: No market demand
- **Viability < 30**: Not a sustainable business
- **Viability < 50 AND Desirability < 40**: Weak on both fronts
- **Feasibility < 25**: Impossible to build
- **Feasibility < 40 AND (Viability < 50 OR Desirability < 50)**: Multiple weaknesses

## 🔍 What Gets Analyzed

### Desirability
- TAM size and growth
- Customer pain level and urgency
- Willingness to pay
- Competitive satisfaction gaps
- Switching costs to new solution
- Problem severity vs. solution capability

### Viability
- Total addressable customers
- Funding model fit (Bootstrap vs. VC)
- Pricing sustainability
- Unit economics (CAC, LTV, payback)
- Churn and retention benchmarks
- Referral loop viability
- Go-to-market feasibility

### Feasibility
- Tech stack and MVP scope
- Build timeline to first revenue
- Team requirements and hiring
- 10x growth achievability
- Product-market fit indicators
- Go-to-market execution capability

## 📚 Requirements

- Python 3.10+
- LangChain
- LangGraph
- Groq Python Client
- Tavily Search API access
- python-dotenv
- pydantic

## 🤝 Contributing

Contributions welcome! Areas for enhancement:
- MCP (Model Context Protocol) integration
- Additional search providers
- Export to PDF/docs
- Benchmark comparisons
- Historical tracking

## 📄 License

MIT

## 🙋 Support

For issues, questions, or feature requests, please open an issue on GitHub.

---

**Built with:**
- 🤖 Groq ChatGroq (Qwen 3.6 27B)
- 🔗 LangChain + LangGraph
- 🔍 Tavily Search API
- 🐍 Python
