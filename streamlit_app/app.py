
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page configuration
st.set_page_config(
    page_title="Olist Retention Predictor",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {padding: 2rem;}
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button:hover {background-color: #27ae60;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained Logistic Regression model"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    model_path = os.path.join(project_root, 'outputs', 'models', 'logistic_regression_final.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ============================================================================
# HEADER
# ============================================================================

st.title("🛒 Olist Customer Retention Predictor")
st.markdown("### Predict first-time customer drop-off risk")

st.markdown("""
**Problem:** 95% of Olist first-time customers never make a second purchase.

**Solution:** This tool predicts which customers are at risk of dropping off 
based on their first order characteristics, enabling targeted retention interventions.

**How it works:**
1. Enter customer's first order details below
2. Click "Predict Drop-off Risk"
3. Get instant risk assessment and recommendations
""")

st.divider()

# ============================================================================
# INPUT FORM
# ============================================================================

st.header("📝 Enter Customer First Order Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🚚 Delivery Info")
    delivery_delay = st.number_input(
        "Delivery Delay (days)",
        min_value=-30,
        max_value=100,
        value=0,
        help="Negative = early, Positive = late"
    )
    
    days_to_delivery = st.number_input(
        "Total Days to Delivery",
        min_value=0,
        max_value=100,
        value=10
    )

with col2:
    st.subheader("💰 Order Economics")
    freight_pct = st.slider(
        "Freight % of Order Value",
        min_value=0.0,
        max_value=80.0,
        value=15.0,
        help="Shipping cost as % of order value"
    )
    
    num_items = st.number_input(
        "Number of Items",
        min_value=1,
        max_value=20,
        value=1
    )
    
    price_per_item = st.number_input(
        "Price per Item (R$)",
        min_value=1.0,
        max_value=10000.0,
        value=100.0
    )
    
    uses_installments = st.checkbox("Uses Installment Payment", value=False)

with col3:
    st.subheader("👤 Customer & Product")
    is_southeast = st.checkbox(
        "Southeast Brazil Customer",
        value=True,
        help="SP, RJ, MG, ES states"
    )
    
    is_repeatable_category = st.checkbox(
        "Repeatable Category",
        value=False,
        help="Health/beauty, books, pet supplies"
    )
    
    is_heavy_product = st.checkbox("Heavy Product (>5kg)", value=False)
    
    has_comment = st.checkbox("Left Review Comment", value=False)
    
    is_holiday_season = st.checkbox(
        "Holiday Season Purchase",
        value=False,
        help="November or December"
    )
    
    is_weekend = st.checkbox("Weekend Purchase", value=False)

st.divider()

# ============================================================================
# ADVANCED OPTIONS (COLLAPSED)
# ============================================================================

with st.expander("⚙️ Advanced Options (Optional)"):
    adv_col1, adv_col2 = st.columns(2)
    
    with adv_col1:
        purchase_month = st.selectbox(
            "Purchase Month",
            options=list(range(1, 13)),
            index=10  # November default
        )
        
        payment_type = st.radio(
            "Payment Type",
            options=["Credit Card", "Boleto", "Debit Card"],
            index=0
        )
    
    with adv_col2:
        cluster = st.selectbox(
            "Customer Segment",
            options=["Unknown", "Budget Shoppers", "High Freight/Risk"],
            index=0
        )
        
        state = st.selectbox(
            "Customer State",
            options=["SP", "RJ", "MG", "RS", "PR", "SC", "BA", "DF", "ES", "GO", "Other"],
            index=0
        )

# ============================================================================
# PREDICTION
# ============================================================================

if st.button(" Predict Drop-off Risk", type="primary"):
    
    # Calculate derived features
    is_late_delivery = int(delivery_delay > 0)
    is_very_late = int(delivery_delay > 10)
    is_early_delivery = int(delivery_delay < 0)
    is_high_freight = int(freight_pct > 20)
    
    # Payment type one-hot encoding
    payment_boleto = int(payment_type == "Boleto")
    payment_credit_card = int(payment_type == "Credit Card")
    payment_debit_card = int(payment_type == "Debit Card")
    
    # Cluster encoding
    cluster_0 = int(cluster == "Budget Shoppers")
    cluster_1 = int(cluster == "High Freight/Risk")
    
    # State encoding
    state_map = {
        'BA': 'state_BA', 'DF': 'state_DF', 'ES': 'state_ES',
        'GO': 'state_GO', 'MG': 'state_MG', 'PR': 'state_PR',
        'RJ': 'state_RJ', 'RS': 'state_RS', 'SC': 'state_SC',
        'SP': 'state_SP'
    }
    
    state_features = {
        'state_BA': 0, 'state_DF': 0, 'state_ES': 0, 'state_GO': 0,
        'state_MG': 0, 'state_PR': 0, 'state_RJ': 0, 'state_RS': 0,
        'state_SC': 0, 'state_SP': 0
    }
    
    if state in state_map:
        state_features[state_map[state]] = 1
    
    # Day of week calculation
    purchase_day_of_week = 5 if is_weekend else 2  # Saturday or Tuesday
    
    # Create feature vector matching training data (33 features)
    features = pd.DataFrame({
        'delivery_delay': [delivery_delay],
        'is_late_delivery': [is_late_delivery],
        'is_very_late': [is_very_late],
        'is_early_delivery': [is_early_delivery],
        'freight_pct': [freight_pct],
        'is_high_freight': [is_high_freight],
        'num_items': [num_items],
        'price_per_item': [price_per_item],
        'uses_installments': [int(uses_installments)],
        'is_southeast': [int(is_southeast)],
        'is_repeatable_category': [int(is_repeatable_category)],
        'is_heavy_product': [int(is_heavy_product)],
        'has_comment': [int(has_comment)],
        'purchase_month': [purchase_month],
        'purchase_day_of_week': [purchase_day_of_week],
        'is_weekend': [int(is_weekend)],
        'is_holiday_season': [int(is_holiday_season)],
        'days_to_delivery': [days_to_delivery],
        'cluster_0': [cluster_0],
        'cluster_1': [cluster_1],
        'payment_boleto': [payment_boleto],
        'payment_credit_card': [payment_credit_card],
        'payment_debit_card': [payment_debit_card],
        'state_BA': [state_features['state_BA']],
        'state_DF': [state_features['state_DF']],
        'state_ES': [state_features['state_ES']],
        'state_GO': [state_features['state_GO']],
        'state_MG': [state_features['state_MG']],
        'state_PR': [state_features['state_PR']],
        'state_RJ': [state_features['state_RJ']],
        'state_RS': [state_features['state_RS']],
        'state_SC': [state_features['state_SC']],
        'state_SP': [state_features['state_SP']]
    })
    
    try:
        # Make prediction
        prediction_proba = model.predict_proba(features)[0]
        drop_off_prob = prediction_proba[1] * 100  # Convert to percentage
        retention_prob = prediction_proba[0] * 100
        
        # Determine risk level
        if drop_off_prob >= 98:
            risk_level = "🔴 CRITICAL RISK"
            risk_color = "red"
        elif drop_off_prob >= 95:
            risk_level = "🟠 HIGH RISK"
            risk_color = "orange"
        elif drop_off_prob >= 90:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "blue"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
        
        # Display results
        st.divider()
        st.header("Prediction Results")
        
        # Metrics row
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Drop-off Probability",
                f"{drop_off_prob:.1f}%",
                delta=f"{drop_off_prob - 95:.1f}% vs baseline",
                delta_color="inverse"
            )
        
        with metric_col2:
            st.metric(
                "Retention Probability",
                f"{retention_prob:.1f}%",
                delta=f"{retention_prob - 5:.1f}% vs baseline"
            )
        
        with metric_col3:
            if risk_color == "red":
                st.error(risk_level)
            elif risk_color == "orange":
                st.warning(risk_level)
            elif risk_color == "blue":
                st.info(risk_level)
            else:
                st.success(risk_level)
        
        # Recommendations
        st.subheader("💡 Personalized Recommendations")
        
        recommendations = []
        
        # Risk-specific recommendations
        if drop_off_prob >= 95:
            recommendations.append("🚨 **URGENT:** High-risk customer - activate retention protocol immediately")
            recommendations.append("📧 Send personalized follow-up email within 24 hours")
            recommendations.append("🎁 Offer 20% discount coupon valid for 7 days")
        
        # Feature-specific recommendations
        if freight_pct > 20:
            recommendations.append("📦 **High shipping cost detected** - Consider free shipping offer or subsidy")
        
        if not is_repeatable_category:
            recommendations.append("🔄 **Non-repeatable product** - Focus on cross-sell to recurring categories")
        
        if not uses_installments:
            recommendations.append("💳 **Promote installment payments** - Historical data shows 26% retention increase")
        
        if is_holiday_season:
            recommendations.append("✅ **Holiday purchase advantage** - These customers are 61% more likely to return!")
            recommendations.append("🎄 Send targeted holiday follow-up campaign")
        else:
            recommendations.append("📅 Consider seasonal promotion to re-engage")
        
        if is_late_delivery:
            recommendations.append("⏰ **Late delivery detected** - Issue apology credit or compensation")
        
        if not is_southeast:
            recommendations.append("🗺️ **Non-Southeast customer** - Extra attention needed for retention")
        
        # Display recommendations
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # ROI Calculator
        st.subheader("💰 Retention ROI Estimate")
        
        roi_col1, roi_col2 = st.columns(2)
        
        with roi_col1:
            st.markdown("**Intervention Cost:** R$ 15 per customer")
            st.markdown("**Expected Success Rate:** 30%")
            st.markdown("**Customer LTV (if retained):** R$ 160")
        
        with roi_col2:
            expected_value = (retention_prob / 100) * 160 - 15
            roi = ((expected_value + 15) / 15 - 1) * 100 if expected_value > -15 else -100
            
            st.metric("Expected Value", f"R$ {expected_value:.2f}")
            st.metric("ROI", f"{roi:.1f}%")
            
            if expected_value > 0:
                st.success("✅ Intervention recommended - Positive ROI expected")
            else:
                st.warning("⚠️ Intervention not cost-effective at current probability")
        
        # Success - prediction complete
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        st.error("Please check that all inputs are valid and try again.")

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p><strong>Olist Marketplace Integrity Audit</strong> | Developed by Reynold Choruma | March 2026</p>
    <p>Model: Logistic Regression | PR AUC: 0.9654 | Minority Class Recall: 62.6%</p>
</div>
""", unsafe_allow_html=True)
