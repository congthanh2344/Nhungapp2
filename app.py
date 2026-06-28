import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==============================================================================
# 1. CẤU HÌNH HỆ THỐNG VÀ GIAO DIỆN
# ==============================================================================
st.set_page_config(
    page_title="Portfolio Optimization Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Áp dụng giao diện thống nhất cho đồ thị Matplotlib
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['figure.facecolor'] = '#ffffff'
plt.rcParams['axes.facecolor'] = '#ffffff'

# ==============================================================================
# 2. ĐỊNH NGHĨA CÁC HÀM ĐỊNH LƯỢNG (CORE LOGIC)
# ==============================================================================
def optimize_markowitz(rets, rf_rate):
    """
    Tối ưu hóa danh mục theo mô hình Markowitz để tìm Sharpe Ratio lớn nhất.
    Tỷ trọng mỗi cổ phiếu bị giới hạn trong khoảng [10%, 80%].
    """
    if rets.empty or len(rets.columns) < 1: 
        return np.array([])
    
    mean_rets = rets.mean() * 250
    cov_matrix = rets.cov() * 250
    n = len(rets.columns)
    
    def min_sharpe(w):
        p_ret = np.sum(mean_rets * w)
        p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        return -(p_ret - rf_rate) / p_vol if p_vol > 0 else 0

    # Ràng buộc: Tổng tỷ trọng bằng 1 (100%)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    # Giới hạn tỷ trọng từ 10% đến 80% cho mỗi cổ phiếu
    bnds = tuple((0.1, 0.8) for _ in range(n))
    
    res = minimize(min_sharpe, [1/n]*n, method='SLSQP', bounds=bnds, constraints=cons)
    return res.x

def get_period_stats(rets, bench_rets, rf_rate):
    """
    Tính toán lợi nhuận tích lũy và rủi ro sụt giảm lớn nhất (Max Drawdown) cho từng giai đoạn.
    """
    if rets.empty or bench_rets.empty:
        return {'LN Tích lũy (%)': 0, 'MDD (%)': 0, 'LN VN-Index (%)': 0, 'MDD VN-Index (%)': 0}
        
    cum = (1 + rets).cumprod()
    b_cum = (1 + bench_rets).cumprod()
    
    res = {
        'LN Tích lũy (%)': (cum.iloc[-1] - 1) * 100,
        'MDD (%)': ((cum / cum.cummax() - 1).min()) * 100,
        'LN VN-Index (%)': (b_cum.iloc[-1] - 1) * 100,
        'MDD VN-Index (%)': ((b_cum / b_cum.cummax() - 1).min()) * 100
    }
    return res

# ==============================================================================
# 3. GIAO DIỆN CHÍNH & SIDEBAR THAM SỐ
# ==============================================================================
st.title("📊 Ứng dụng Tối ưu hóa & Kiểm thử Danh mục Đầu tư")
st.markdown("Xây dựng chiến lược phân bổ tài sản động kết hợp **Xung lực dòng tiền (Momentum)** ngắn hạn và **Lý thuyết hiện đại Markowitz**.")
st.hr()

# --- SIDEBAR CẤU HÌNH ---
st.sidebar.header("⚙️ Tham số Chiến lược")

# Tải file CSV dữ liệu đầu vào
uploaded_file = st.sidebar.file_uploader("Tải lên file dữ liệu HOSE (CSV)", type=["csv"])

# Các thông số tùy chỉnh chiến lược
stop_loss = st.sidebar.slider("Ngưỡng Stop Loss hàng ngày", -0.30, -0.05, -0.15, 0.01)
rf_rate = st.sidebar.number_input("Lãi suất phi rủi ro (Risk-free Rate)", value=0.045, step=0.005, format="%.3f")
top_n = st.sidebar.number_input("Số lượng cổ phiếu nắm giữ trong danh mục", min_value=1, max_value=10, value=3)

# Lựa chọn tháng tái cân bằng (Rebalance)
st.sidebar.markdown("**Tháng tái cân bằng**")
m_jan = st.sidebar.checkbox("Tháng 1 (Q1)", value=True)
m_apr = st.sidebar.checkbox("Tháng 4 (Q2)", value=True)
m_jul = st.sidebar.checkbox("Tháng 7 (Q3)", value=True)
m_oct = st.sidebar.checkbox("Tháng 10 (Q4)", value=True)

rebalance_months = []
if m_jan: rebalance_months.append(1)
if m_apr: rebalance_months.append(4)
if m_jul: rebalance_months.append(7)
if m_oct: rebalance_months.append(10)

# ==============================================================================
# 4. XỬ LÝ DỮ LIỆU VÀ QUY TRÌNH BACKTEST
# ==============================================================================
if uploaded_file is not None:
    try:
        # Đọc dữ liệu
        df = pd.read_csv(uploaded_file, low_memory=False)
        df['date'] = pd.to_datetime(df['date'])
        
        # Biến đổi cấu trúc bảng (Pivot) dữ liệu giá đóng cửa
        price_matrix = df.pivot(index='date', columns='ticker', values='close').ffill().dropna(axis=1, thresh=500).dropna()
        all_rets = price_matrix.pct_change().dropna()
        
        # Tách tập dữ liệu kiểm thử (Từ năm 2023)
        test_rets = all_rets.loc['2023-01-01':]
        vni_proxy = test_rets.mean(axis=1) # VN-Index proxy (Trung bình thị trường)
        
        if test_rets.empty:
            st.error("❌ Dữ liệu tải lên không chứa mốc thời gian từ năm 2023 trở đi để tiến hành backtest.")
        elif len(rebalance_months) == 0:
            st.warning("⚠️ Vui lòng chọn ít nhất một tháng tái cân bằng ở thanh Sidebar.")
        else:
            # --- 4.1 Thực thi Chiến lược Động (Dynamic Strategy) ---
            dynamic_rets_list = []
            curr_tickers, curr_w = [], []
            rebalance_log = []
            
            for d in test_rets.index:
                # Điều kiện tái cân bằng: Thuộc tháng quy định và là ngày giao dịch đầu tiên của tháng đó
                if d.month in rebalance_months and d == test_rets.index[test_rets.index.month == d.month][0]:
                    # Lấy dữ liệu 6 tháng trước đó để tính toán Momentum
                    lb_data = all_rets.loc[d - pd.DateOffset(months=6) : d - pd.Timedelta(days=1)]
                    
                    if not lb_data.empty:
                        # Tính hiệu suất lợi nhuận (Momentum) 6 tháng qua
                        mom = (1 + lb_data).prod() - 1
                        # Chọn Top N mã tăng trưởng mạnh nhất
                        curr_tickers = mom.sort_values(ascending=False).head(top_n).index.tolist()
                        # Tối ưu hóa tỷ trọng phân bổ danh mục qua mô hình Markowitz
                        curr_w = optimize_markowitz(lb_data[curr_tickers], rf_rate)
                        
                        # Ghi nhật ký lịch sử kỳ rebalance
                        rebalance_log.append({
                            'Ngày': d.date().strftime('%Y-%m-%d'),
                            'Danh mục': ', '.join(curr_tickers),
                            'Tỷ trọng': ', '.join([f'{w:.1%}' for w in curr_w])
                        })
                
                # Tính lợi nhuận thực tế hàng ngày dựa trên danh mục hiện tại
                if len(curr_tickers) > 0:
                    day_ret = np.sum(test_rets.loc[d, curr_tickers] * curr_w)
                else:
                    day_ret = 0
                
                # Áp dụng cơ chế dừng lỗ hàng ngày (Daily Stop Loss)
                if day_ret < stop_loss: 
                    day_ret = stop_loss
                    
                dynamic_rets_list.append(day_ret)
                
            dynamic_rets = pd.Series(dynamic_rets_list, index=test_rets.index)
            
            # --- 4.2 Tính toán các Chiến lược Tĩnh làm Benchmark ---
            train_2022 = all_rets.loc['2022-01-01':'2022-12-31']
            
            if not train_2022.empty:
                # Chọn top mã dựa trên hiệu suất năm 2022 làm danh mục tĩnh cho năm 2023
                static_tickers = ((1 + train_2022).prod() - 1).sort_values(ascending=False).head(top_n).index.tolist()
                # Chiến lược Static Equal Weight (Chia đều tỷ trọng)
                ew_static = test_rets[static_tickers].mean(axis=1)
                # Chiến lược Static Buy & Hold (Tối ưu Markowitz 1 lần từ đầu năm và giữ nguyên)
                static_w = optimize_markowitz(train_2022[static_tickers], rf_rate)
                bh_static = (test_rets[static_tickers] * static_w).sum(axis=1)
            else:
                # Trường hợp dữ liệu quá khứ năm 2022 bị khuyết
                ew_static = vni_proxy * 0
                bh_static = vni_proxy * 0
            
            # Hợp nhất kết quả tăng trưởng tích lũy của các chiến lược
            results_df = pd.DataFrame({
                'Dynamic (Opt+SL)': (1 + dynamic_rets).cumprod(),
                'VN-Index': (1 + vni_proxy).cumprod(),
                'Static EW': (1 + ew_static).cumprod(),
                'Static B&H': (1 + bh_static).cumprod()
            })
            
            # ==============================================================================
            # 5. HIỂN THỊ KẾT QUẢ TRÊN DASHBOARD
            # ==============================================================================
            
            # Kính 1: Đồ thị tăng trưởng tổng quan
            st.subheader("📈 Đồ thị Tăng trưởng Tài sản Lũy kế (2023)")
            fig, ax = plt.subplots(figsize=(14, 6))
            results_df.plot(ax=ax, linewidth=2.5)
            ax.set_title('Tổng hợp So sánh Chiến lược 2023', fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel('Thời gian', fontsize=11)
            ax.set_ylabel('Giá trị tài sản tích lũy (Gốc = 1.0)', fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(frameon=True, facecolor='#ffffff', edgecolor='none')
            st.pyplot(fig)
            plt.close()
            
            # Kính 2: Bảng tổng hợp hiệu quả chi tiết 2023
            st.subheader("📊 Bảng Tổng hợp Hiệu quả Kỹ thuật")
            summary = []
            for col in results_df.columns:
                # Phân định lại chuỗi lợi nhuận đơn lẻ phục vụ tính Sharpe Ratio
                rets = dynamic_rets if 'Dynamic' in col else (vni_proxy if 'VN' in col else (ew_static if 'EW' in col else bh_static))
                cum_series = results_df[col]
                
                std_dev = rets.std() * np.sqrt(250)
                sharpe_val = (rets.mean() * 250 - rf_rate) / std_dev if std_dev > 0 else 0
                max_dd = (cum_series / cum_series.cummax() - 1).min()
                
                summary.append({
                    'Chiến lược': col,
                    'Lợi nhuận 2023': f'{(cum_series.iloc[-1] - 1):.2%}',
                    'Max Drawdown': f'{max_dd:.2%}',
                    'Hệ số Sharpe': f'{sharpe_val:.2f}'
                })
            st.table(pd.DataFrame(summary))
            
            # Kính 3: Phân tích sâu theo chu kỳ/giai đoạn thị trường (Market Regimes)
            st.subheader("🔍 Hiệu quả Chiến lược theo Giai đoạn Thị trường")
            regimes = {
                'Uptrend (T1-T8)': ('2023-01-01', '2023-08-31'),
                'Downtrend (T9-T10)': ('2023-09-01', '2023-10-31'),
                'Sideway (T11-T12)': ('2023-11-01', '2023-12-31')
            }
            
            regime_data = []
            for name, (start, end) in regimes.items():
                p_rets = dynamic_rets.loc[start:end]
                p_vni = vni_proxy.loc[start:end]
                if not p_rets.empty:
                    stats = get_period_stats(p_rets, p_vni, rf_rate)
                    stats['Giai đoạn'] = name
                    regime_data.append(stats)
            
            if regime_data:
                df_regime = pd.DataFrame(regime_data)
                df_regime = df_regime[['Giai đoạn', 'LN Tích lũy (%)', 'LN VN-Index (%)', 'MDD (%)', 'MDD VN-Index (%)']]
                
                # Vẽ đồ thị so sánh lợi nhuận và rủi ro theo từng giai đoạn
                col1, col2 = st.columns(2)
                with col1:
                    fig_l, ax_l = plt.subplots(figsize=(6, 4))
                    df_regime.plot(x='Giai đoạn', y=['LN Tích lũy (%)', 'LN VN-Index (%)'], kind='bar', ax=ax_l, color=['#7B1FA2', '#E53935'])
                    ax_l.set_title('Lợi nhuận theo Giai đoạn (%)', fontsize=12, fontweight='bold')
                    ax_l.set_ylabel('%')
                    ax_l.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.xticks(rotation=0)
                    st.pyplot(fig_l)
                    plt.close()
                with col2:
                    fig_r, ax_r = plt.subplots(figsize=(6, 4))
                    df_regime.plot(x='Giai đoạn', y=['MDD (%)', 'MDD VN-Index (%)'], kind='bar', ax=ax_r, color=['#BA68C8', '#FF8A80'])
                    ax_r.set_title('Max Drawdown theo Giai đoạn (%)', fontsize=12, fontweight='bold')
                    ax_r.set_ylabel('%')
                    ax_r.grid(axis='y', linestyle='--', alpha=0.5)
                    plt.xticks(rotation=0)
                    st.pyplot(fig_r)
                    plt.close()
                
                # Hiện bảng dữ liệu định dạng số đẹp
                st.dataframe(df_regime.style.format({
                    'LN Tích lũy (%)': '{:.2f}%',
                    'LN VN-Index (%)': '{:.2f}%',
                    'MDD (%)': '{:.2f}%',
                    'MDD VN-Index (%)': '{:.2f}%'
                }), use_container_width=True)
            
            # Kính 4: Nhật ký cấu trúc danh mục qua các kỳ Tái cân bằng
            st.subheader("📅 Chi tiết Danh mục & Tỷ trọng phân bổ qua từng kỳ Rebalance")
            if rebalance_log:
                st.dataframe(pd.DataFrame(rebalance_log), use_container_width=True)
            else:
                st.info("Không có sự kiện tái cân bằng nào xảy ra trong phạm vi tham số đã chọn.")
                
    except Exception as e:
        st.error(f"❌ Đã xảy ra lỗi trong quá trình biên dịch dữ liệu: {e}")
        st.info("Vui lòng kiểm tra lại cấu trúc file CSV. File hợp lệ bắt buộc phải gồm cấu trúc cột tiêu đề: 'date', 'ticker' và 'close'.")
else:
    st.info("💡 Mẹo: Hãy bắt đầu bằng việc tải lên file dữ liệu giá lịch sử dạng `.csv` ở thanh menu bên trái.")