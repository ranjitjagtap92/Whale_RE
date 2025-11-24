import streamlit as st

st.write("🔍 Debug Mode - Testing imports...")

try:
    st.write("✅ Streamlit working")
    import pandas as pd
    st.write("✅ Pandas working")
    import orchestrator
    st.write("✅ Orchestrator working")
    import utils
    st.write("✅ Utils working")
    import agents
    st.write("✅ Agents working")
    st.write("✅ All imports successful! The issue is likely during app initialization.")
except Exception as e:
    st.error(f"❌ Import failed: {e}")
    import traceback
    st.code(traceback.format_exc())
