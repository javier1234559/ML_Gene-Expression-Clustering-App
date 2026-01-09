import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.cluster.hierarchy import linkage, fcluster
import time

# ============================================================================
# CLUSTERING ALGORITHMS LAYER
# ============================================================================

# K-MEANS
def km_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def km_init(X, K, seed=42):
    np.random.seed(seed)
    return X[np.random.choice(X.shape[0], K, replace=False)].copy()

def km_assign(X, centroids):
    return np.array([np.argmin([km_euclidean(x, c) for c in centroids]) for x in X])

def km_update(X, labels, K):
    return np.array([X[labels==k].mean(axis=0) if np.sum(labels==k)>0 else np.zeros(X.shape[1]) for k in range(K)])

def km_fit(X, K, max_iter=100, seed=42):
    cents = km_init(X, K, seed)
    for i in range(max_iter):
        old = cents.copy()
        labs = km_assign(X, cents)
        cents = km_update(X, labs, K)
        if np.allclose(cents, old, atol=1e-6):
            return labs, cents
    return labs, cents

# AGGLOMERATIVE
def agg_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def agg_average_linkage(cluster1, cluster2, X):
    dists = [agg_euclidean(X[i], X[j]) for i in cluster1 for j in cluster2]
    return np.mean(dists)

def agg_fit(X, K):
    clusters = [[i] for i in range(len(X))]
    while len(clusters) > K:
        min_dist = float('inf')
        pair = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = agg_average_linkage(clusters[i], clusters[j], X)
                if dist < min_dist:
                    min_dist = dist
                    pair = (i, j)
        i, j = pair
        new_cluster = clusters[i] + clusters[j]
        clusters.pop(j)
        clusters.pop(i)
        clusters.append(new_cluster)
    labels = np.zeros(len(X), dtype=int)
    for idx, cluster in enumerate(clusters):
        for point_idx in cluster:
            labels[point_idx] = idx
    return labels

# SPECTRAL
def spec_euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def spec_affinity_matrix(X, sigma=1.0):
    n = X.shape[0]
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = spec_euclidean(X[i], X[j])
            W[i, j] = W[j, i] = np.exp(-dist**2 / (2 * sigma**2))
    return W

def spec_fit(X, K, sigma=10.0, seed=42):
    W = spec_affinity_matrix(X, sigma)
    D = np.diag(W.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))
    L = np.eye(len(X)) - D_inv_sqrt @ W @ D_inv_sqrt
    eigenvals, eigenvecs = np.linalg.eigh(L)
    X_embedded = eigenvecs[:, :K]
    np.random.seed(seed)
    cents = X_embedded[np.random.choice(X_embedded.shape[0], K, replace=False)]
    for _ in range(100):
        old = cents.copy()
        dists = np.array([[spec_euclidean(x, c) for c in cents] for x in X_embedded])
        labels = np.argmin(dists, axis=1)
        cents = np.array([X_embedded[labels==k].mean(axis=0) if np.sum(labels==k)>0 else np.zeros(K) for k in range(K)])
        if np.allclose(cents, old, atol=1e-6):
            break
    return labels

# ENSEMBLE
def ensemble_fit(X, K, labels_km, labels_agg, labels_spec):
    n_samples = X.shape[0]
    consensus_matrix = np.zeros((n_samples, n_samples))
    
    all_labels = [labels_km, labels_agg, labels_spec]
    for labels in all_labels:
        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1
    
    consensus_matrix /= len(all_labels)
    distance_matrix = 1 - consensus_matrix
    linkage_matrix = linkage(distance_matrix[np.triu_indices(n_samples, k=1)], method='average')
    labels_ensemble = fcluster(linkage_matrix, K, criterion='maxclust') - 1
    
    return labels_ensemble, consensus_matrix

# ============================================================================
# DATA PREPROCESSING LAYER
# ============================================================================

@st.cache_data(show_spinner=False)
def preprocess_data(df_hash):
    df = st.session_state.data
    
    # Select only numeric columns
    X = df.select_dtypes(include=[np.number])
    
    # Remove AFFX probes if exist
    X = X[[c for c in X.columns if not str(c).startswith('AFFX')]]
    
    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA
    pca = PCA(n_components=0.90, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    return X, X_scaled, X_pca, pca

@st.cache_data(show_spinner=False)
def run_clustering(_X_pca, K, method):
    start_time = time.time()
    
    if method == "K-Means":
        labels, _ = km_fit(_X_pca, K)
        extra_data = None
    elif method == "Agglomerative":
        labels = agg_fit(_X_pca, K)
        extra_data = None
    elif method == "Spectral":
        labels = spec_fit(_X_pca, K)
        extra_data = None
    elif method == "Ensemble":
        labels_km, _ = km_fit(_X_pca, K)
        labels_agg = agg_fit(_X_pca, K)
        labels_spec = spec_fit(_X_pca, K)
        labels, consensus = ensemble_fit(_X_pca, K, labels_km, labels_agg, labels_spec)
        extra_data = consensus
    
    elapsed_time = time.time() - start_time
    return labels, elapsed_time, extra_data

# ============================================================================
# VISUALIZATION LAYER
# ============================================================================

def create_cluster_plot(X_pca, labels, K, method):
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#34495e']
    
    fig = go.Figure()
    
    # Plot each cluster
    for k in range(K):
        mask = labels == k
        fig.add_trace(
            go.Scatter(
                x=X_pca[mask, 0],
                y=X_pca[mask, 1],
                mode='markers',
                name=f'Cluster {k+1}',
                marker=dict(size=12, color=colors[k % len(colors)], opacity=0.7, 
                           line=dict(width=1, color='white')),
            )
        )
    
    fig.update_layout(
        title=f'{method} Clustering (K={K})',
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=600,
        template='plotly_white',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig

def create_consensus_heatmap(consensus_matrix):
    fig = go.Figure(data=go.Heatmap(
        z=consensus_matrix,
        colorscale='YlOrRd',
        colorbar=dict(title="Consensus")
    ))
    
    fig.update_layout(
        title="Consensus Matrix",
        xaxis_title="Sample",
        yaxis_title="Sample",
        height=450,
        template='plotly_white'
    )
    
    return fig

def create_cluster_summary(labels, K, X):
    summary = []
    
    for k in range(K):
        mask = labels == k
        count = np.sum(mask)
        percentage = (count / len(labels)) * 100
        
        cluster_data = X.iloc[mask]
        mean_values = cluster_data.mean()
        top_features = mean_values.nlargest(3)
        
        summary.append({
            'cluster': k + 1,
            'count': count,
            'percentage': percentage,
            'top_features': top_features
        })
    
    return summary

def analyze_consensus_matrix(consensus_matrix):
    n = consensus_matrix.shape[0]
    upper_indices = np.triu_indices(n, k=1)
    consensus_values = consensus_matrix[upper_indices]
    
    total_pairs = len(consensus_values)
    strong_agreement = np.sum(consensus_values >= 0.67)
    weak_agreement = np.sum(consensus_values <= 0.33)
    mixed_agreement = total_pairs - strong_agreement - weak_agreement
    
    avg_consensus = np.mean(consensus_values)
    
    return {
        'total_pairs': total_pairs,
        'strong_agreement': strong_agreement,
        'weak_agreement': weak_agreement,
        'mixed_agreement': mixed_agreement,
        'avg_consensus': avg_consensus,
        'strong_pct': (strong_agreement / total_pairs) * 100,
        'weak_pct': (weak_agreement / total_pairs) * 100,
        'mixed_pct': (mixed_agreement / total_pairs) * 100
    }

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    st.title("Ứng dụng Clustering dữ liệu biểu hiện Gen")
    st.caption("K-Means | Agglomerative | Spectral | Ensemble")
    st.divider()

def render_data_source():
    st.subheader("Nguồn dữ liệu")
    
    tab1, tab2 = st.tabs(["Tải lên CSV", "Dữ liệu mẫu"])
    
    with tab1:
        file = st.file_uploader("Tải lên file CSV với dữ liệu số", type=['csv'], label_visibility="collapsed")
        if file:
            df = pd.read_csv(file)
            st.session_state.data = df
            st.session_state.data_loaded = True
            
            # Pre-process immediately
            df_hash = (df.shape, id(df))
            with st.spinner("Preprocessing data..."):
                X, X_scaled, X_pca, pca = preprocess_data(df_hash)
                st.session_state.X = X
                st.session_state.X_pca = X_pca
                st.session_state.pca = pca
            
            st.success(f"Đã tải: {len(df)} mẫu × {len(df.columns)} đặc trưng")
    
    with tab2:
        if st.button("Tải dữ liệu Leukemia", use_container_width=True):
            try:
                df = pd.read_csv('Leukemia.csv')
                # Drop type column if exists (unsupervised!)
                if 'type' in df.columns:
                    df = df.drop(columns=['type'])
                if 'samples' in df.columns:
                    df = df.drop(columns=['samples'])
                st.session_state.data = df
                st.session_state.data_loaded = True
                
                # Pre-process immediately
                df_hash = (df.shape, id(df))
                with st.spinner("Preprocessing data..."):
                    X, X_scaled, X_pca, pca = preprocess_data(df_hash)
                    st.session_state.X = X
                    st.session_state.X_pca = X_pca
                    st.session_state.pca = pca
                
                st.success(f"Đã tải: {len(df)} mẫu × {len(df.columns)} đặc trưng")
            except FileNotFoundError:
                st.error("Không tìm thấy file Leukemia.csv trong thư mục hiện tại")

def render_data_preview(df):
    st.divider()
    st.subheader("Xem trước dữ liệu")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    col1, col2 = st.columns(2)
    col1.metric("Số mẫu", len(df))
    col2.metric("Số đặc trưng", len(numeric_cols))
    
    # Show PCA status if already preprocessed
    if st.session_state.get('data_loaded') and st.session_state.get('X_pca') is not None:
        st.success(f"Sẵn sàng: {st.session_state.X_pca.shape[1]} thành phần PCA (90% phương sai)")
    
    with st.expander("Xem dữ liệu", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=300)

def render_model_config():
    st.subheader("Cấu hình")
    
    # K slider - with key to prevent unnecessary reruns
    K = st.slider("Số cụm (K):", 2, 10, 3, key="num_clusters", 
                  help="Chọn số cụm muốn phát hiện trong dữ liệu")
    
    method = st.selectbox(
        "Phương pháp Clustering:",
        ["K-Means", "Agglomerative", "Spectral", "Ensemble"],
        key="selected_method",
        help="K-Means: Nhanh | Agglomerative: Phân cấp | Spectral: Phi tuyến | Ensemble: Kết hợp"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Chạy Clustering", type="primary", use_container_width=True):
            with st.spinner(f"Đang chạy {method} với K={K}..."):
                # Use pre-processed data (already in session state)
                X_pca = st.session_state.X_pca
                
                # Run clustering (cached)
                labels, elapsed_time, extra_data = run_clustering(X_pca, K, method)
                
                # Calculate metrics
                sil = silhouette_score(X_pca, labels)
                metrics = {'silhouette': sil, 'time': elapsed_time}
                
                # Create result object for history
                result = {
                    'method': method,
                    'K': K,
                    'labels': labels.copy(),
                    'metrics': metrics.copy(),
                    'extra_data': extra_data,
                    'timestamp': pd.Timestamp.now().strftime("%H:%M:%S")
                }
                
                # Add to history
                if 'history' not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.append(result)
                
                # Store current result
                st.session_state.labels = labels
                st.session_state.K = K
                st.session_state.metrics = metrics
                st.session_state.current_method = method
                st.session_state.extra_data = extra_data
                st.session_state.trained = True
                
            st.rerun()
    
    with col2:
        if st.button("Chạy tất cả", type="secondary", use_container_width=True):
            # Clear old history
            st.session_state.history = []
            
            # Use pre-processed data
            X_pca = st.session_state.X_pca
            
            # Run all 4 methods
            all_methods = ["K-Means", "Agglomerative", "Spectral", "Ensemble"]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, m in enumerate(all_methods):
                status_text.text(f"Đang chạy {m}... ({idx+1}/4)")
                
                # Run clustering
                labels, elapsed_time, extra_data = run_clustering(X_pca, K, m)
                
                # Calculate metrics
                sil = silhouette_score(X_pca, labels)
                metrics = {'silhouette': sil, 'time': elapsed_time}
                
                # Create result object
                result = {
                    'method': m,
                    'K': K,
                    'labels': labels.copy(),
                    'metrics': metrics.copy(),
                    'extra_data': extra_data,
                    'timestamp': pd.Timestamp.now().strftime("%H:%M:%S")
                }
                
                # Add to history
                st.session_state.history.append(result)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(all_methods))
            
            # Store last result (Ensemble) as current
            st.session_state.labels = st.session_state.history[-1]['labels']
            st.session_state.K = K
            st.session_state.metrics = st.session_state.history[-1]['metrics']
            st.session_state.current_method = "Ensemble"
            st.session_state.extra_data = st.session_state.history[-1]['extra_data']
            st.session_state.trained = True
            
            status_text.text("Hoàn thành!")
            st.rerun()


def render_visualization():
    if st.session_state.get('trained'):
        # History tabs
        if len(st.session_state.get('history', [])) > 1:
            st.subheader("Kết quả Clustering")
            
            # Create tabs for history
            history = st.session_state.history
            tab_labels = [f"{h['method'][:4]} K={h['K']} ({h['timestamp']})" for h in history[-5:]]  # Last 5
            tabs = st.tabs(tab_labels)
            
            for idx, tab in enumerate(tabs):
                with tab:
                    result = history[-(len(tabs)-idx)]
                    X_pca = st.session_state.X_pca
                    
                    fig = create_cluster_plot(X_pca, result['labels'], result['K'], result['method'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show metrics inline
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Thời gian", f"{result['metrics']['time']:.2f}s")
                    with col2:
                        st.metric("Silhouette Score", f"{result['metrics']['silhouette']:.4f}")
                    with col3:
                        cluster_sizes = [np.sum(result['labels'] == k) for k in range(result['K'])]
                        st.caption(f"**Kích thước cụm:** {', '.join([str(s) for s in cluster_sizes])}")
                    
                    # Consensus matrix for Ensemble
                    if result['method'] == "Ensemble" and result['extra_data'] is not None:
                        with st.expander("Ma trận Consensus", expanded=False):
                            fig_heatmap = create_consensus_heatmap(result['extra_data'])
                            st.plotly_chart(fig_heatmap, use_container_width=True)
                            
                            st.divider()
                            
                            consensus_analysis = analyze_consensus_matrix(result['extra_data'])
                            
                            st.markdown("**Phân tích độ đồng thuận (Consensus Analysis):**")
                            
                            st.markdown(f"""
                            Ma trận Consensus đo lường mức độ đồng ý giữa 3 thuật toán (K-Means, Agglomerative, Spectral) 
                            khi phân cụm từng cặp mẫu. Giá trị dao động từ 0 (không bao giờ cùng cụm) đến 1 (luôn cùng cụm).
                            """)
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Đồng thuận cao", 
                                         f"{consensus_analysis['strong_agreement']} cặp",
                                         f"{consensus_analysis['strong_pct']:.1f}%")
                                st.caption("≥2/3 thuật toán đồng ý")
                            
                            with col2:
                                st.metric("Đồng thuận trung bình", 
                                         f"{consensus_analysis['mixed_agreement']} cặp",
                                         f"{consensus_analysis['mixed_pct']:.1f}%")
                                st.caption("1/3 < consensus < 2/3")
                            
                            with col3:
                                st.metric("Đồng thuận thấp", 
                                         f"{consensus_analysis['weak_agreement']} cặp",
                                         f"{consensus_analysis['weak_pct']:.1f}%")
                                st.caption("≤1/3 thuật toán đồng ý")
                            
                            st.metric("Độ đồng thuận trung bình", f"{consensus_analysis['avg_consensus']:.3f}")
                            
                            if consensus_analysis['avg_consensus'] >= 0.6:
                                st.success("✓ Độ đồng thuận cao → Kết quả Ensemble tin cậy")
                            elif consensus_analysis['avg_consensus'] >= 0.4:
                                st.warning("⚠ Độ đồng thuận trung bình → Các thuật toán có ý kiến khác nhau")
                            else:
                                st.error("⚠ Độ đồng thuận thấp → Dữ liệu khó phân cụm hoặc K chưa phù hợp")
                    
                    # Báo cáo kết quả phân cụm
                    with st.expander("📊 Báo cáo phân cụm", expanded=True):
                        summary = create_cluster_summary(result['labels'], result['K'], st.session_state.X)
                        
                        st.markdown("**Tổng quan phân cụm:**")
                        
                        for s in summary:
                            st.markdown(f"**Cụm {s['cluster']}**: {s['count']} mẫu ({s['percentage']:.1f}%)")
                            top_feat_str = ', '.join([f'{k}={v:.2f}' for k, v in s['top_features'].items()])
                            st.caption(f"Top 3 đặc trưng: {top_feat_str}")
                        
                        st.divider()
                        
                        largest_cluster = max(summary, key=lambda x: x['count'])
                        smallest_cluster = min(summary, key=lambda x: x['count'])
                        
                        st.markdown("**Nhận xét:**")
                        st.write(f"• Cụm lớn nhất: **Cụm {largest_cluster['cluster']}** với {largest_cluster['count']} mẫu")
                        st.write(f"• Cụm nhỏ nhất: **Cụm {smallest_cluster['cluster']}** với {smallest_cluster['count']} mẫu")
                        
                        imbalance_ratio = largest_cluster['count'] / smallest_cluster['count']
                        if imbalance_ratio > 3:
                            st.write(f"• Phân bố cụm **mất cân bằng** (tỷ lệ {imbalance_ratio:.1f}:1)")
                        else:
                            st.write(f"• Phân bố cụm **tương đối cân bằng** (tỷ lệ {imbalance_ratio:.1f}:1)")
                    
                    # Data preview with cluster labels
                    with st.expander("Xem dữ liệu với nhãn cụm", expanded=False):
                        df_result = pd.DataFrame(st.session_state.X, columns=st.session_state.X.columns)
                        df_result['Cluster'] = result['labels'] + 1
                        st.dataframe(df_result, use_container_width=True, height=400)
                        
                        # Download button for this specific result
                        csv = df_result.to_csv(index=False)
                        st.download_button(
                            "Tải xuống kết quả này",
                            csv,
                            f"{result['method']}_K{result['K']}_{result['timestamp'].replace(':', '-')}.csv",
                            "text/csv",
                            key=f"download_{idx}_{result['timestamp']}",
                            use_container_width=True
                        )
        else:
            # Single result (no tabs)
            st.subheader("Kết quả Clustering")
            X_pca = st.session_state.X_pca
            labels = st.session_state.labels
            K = st.session_state.K
            method = st.session_state.current_method
            
            fig = create_cluster_plot(X_pca, labels, K, method)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional info in columns
            col1, col2 = st.columns(2)
            
            with col1:
                # Show consensus matrix for Ensemble
                if method == "Ensemble" and st.session_state.extra_data is not None:
                    with st.expander("Ma trận Consensus", expanded=False):
                        fig_heatmap = create_consensus_heatmap(st.session_state.extra_data)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        st.divider()
                        
                        consensus_analysis = analyze_consensus_matrix(st.session_state.extra_data)
                        
                        st.markdown("**Phân tích độ đồng thuận (Consensus Analysis):**")
                        
                        st.markdown(f"""
                        Ma trận Consensus đo lường mức độ đồng ý giữa 3 thuật toán (K-Means, Agglomerative, Spectral) 
                        khi phân cụm từng cặp mẫu. Giá trị dao động từ 0 (không bao giờ cùng cụm) đến 1 (luôn cùng cụm).
                        """)
                        
                        col1a, col1b, col1c = st.columns(3)
                        
                        with col1a:
                            st.metric("Đồng thuận cao", 
                                     f"{consensus_analysis['strong_agreement']} cặp",
                                     f"{consensus_analysis['strong_pct']:.1f}%")
                            st.caption("≥2/3 thuật toán đồng ý")
                        
                        with col1b:
                            st.metric("Đồng thuận trung bình", 
                                     f"{consensus_analysis['mixed_agreement']} cặp",
                                     f"{consensus_analysis['mixed_pct']:.1f}%")
                            st.caption("1/3 < consensus < 2/3")
                        
                        with col1c:
                            st.metric("Đồng thuận thấp", 
                                     f"{consensus_analysis['weak_agreement']} cặp",
                                     f"{consensus_analysis['weak_pct']:.1f}%")
                            st.caption("≤1/3 thuật toán đồng ý")
                        
                        st.metric("Độ đồng thuận trung bình", f"{consensus_analysis['avg_consensus']:.3f}")
                        
                        if consensus_analysis['avg_consensus'] >= 0.6:
                            st.success("✓ Độ đồng thuận cao → Kết quả Ensemble tin cậy")
                        elif consensus_analysis['avg_consensus'] >= 0.4:
                            st.warning("⚠ Độ đồng thuận trung bình → Các thuật toán có ý kiến khác nhau")
                        else:
                            st.error("⚠ Độ đồng thuận thấp → Dữ liệu khó phân cụm hoặc K chưa phù hợp")
            
            with col2:
                # PCA info - lazy loading để tránh lag
                if st.checkbox("Xem thông tin PCA", value=False, key="show_pca_info"):
                    pca = st.session_state.pca
                    
                    st.markdown("**Tóm tắt giảm chiều dữ liệu:**")
                    st.write(f"- Số đặc trưng ban đầu: **{st.session_state.X.shape[1]:,}**")
                    st.write(f"- Số thành phần PCA giữ lại: **{X_pca.shape[1]}**")
                    st.write(f"- Tổng phương sai giải thích: **{pca.explained_variance_ratio_.sum()*100:.2f}%**")
                    
                    st.markdown("**Các thành phần chính:**")
                    st.write(f"- PC1 giải thích: **{pca.explained_variance_ratio_[0]*100:.2f}%** phương sai")
                    st.write(f"- PC2 giải thích: **{pca.explained_variance_ratio_[1]*100:.2f}%** phương sai")
                    st.write(f"- PC1+PC2 kết hợp: **{pca.explained_variance_ratio_[:2].sum()*100:.2f}%**")
                    
                    st.markdown("**Tại sao cần PCA?**")
                    st.caption(f"Dữ liệu nhiều chiều ({st.session_state.X.shape[1]:,} đặc trưng) được giảm xuống {X_pca.shape[1]} thành phần trong khi vẫn giữ 90% thông tin. Điều này giúp:")
                    st.caption("• Loại bỏ nhiễu và đặc trưng dư thừa")
                    st.caption("• Tăng tốc độ thuật toán clustering")
                    st.caption("• Cho phép trực quan hóa 2D (PC1 vs PC2)")
                    st.caption("• Giảm độ phức tạp tính toán")
            
            # Báo cáo kết quả phân cụm
            with st.expander("📊 Báo cáo phân cụm", expanded=True):
                summary = create_cluster_summary(labels, K, st.session_state.X)
                
                st.markdown("**Tổng quan phân cụm:**")
                
                for s in summary:
                    st.markdown(f"**Cụm {s['cluster']}**: {s['count']} mẫu ({s['percentage']:.1f}%)")
                    top_feat_str = ', '.join([f'{k}={v:.2f}' for k, v in s['top_features'].items()])
                    st.caption(f"Top 3 đặc trưng: {top_feat_str}")
                
                st.divider()
                
                largest_cluster = max(summary, key=lambda x: x['count'])
                smallest_cluster = min(summary, key=lambda x: x['count'])
                
                st.markdown("**Nhận xét:**")
                st.write(f"• Cụm lớn nhất: **Cụm {largest_cluster['cluster']}** với {largest_cluster['count']} mẫu")
                st.write(f"• Cụm nhỏ nhất: **Cụm {smallest_cluster['cluster']}** với {smallest_cluster['count']} mẫu")
                
                imbalance_ratio = largest_cluster['count'] / smallest_cluster['count']
                if imbalance_ratio > 3:
                    st.write(f"• Phân bố cụm **mất cân bằng** (tỷ lệ {imbalance_ratio:.1f}:1)")
                else:
                    st.write(f"• Phân bố cụm **tương đối cân bằng** (tỷ lệ {imbalance_ratio:.1f}:1)")
            
            # Data preview with cluster labels
            with st.expander("Xem dữ liệu với nhãn cụm", expanded=False):
                df_result = pd.DataFrame(st.session_state.X, columns=st.session_state.X.columns)
                df_result['Cluster'] = labels + 1
                st.dataframe(df_result, use_container_width=True, height=400)
                
                # Download button
                csv = df_result.to_csv(index=False)
                st.download_button(
                    "Tải xuống kết quả này",
                    csv,
                    f"{method}_K{K}_results.csv",
                    "text/csv",
                    use_container_width=True
                )
    else:
        st.info("Cấu hình các tham số và nhấn 'Chạy Clustering'")

def render_export():
    if st.session_state.get('trained'):
        # Clear history button
        if len(st.session_state.get('history', [])) > 1:
            st.divider()
            if st.button("Xóa lịch sử", use_container_width=True):
                st.session_state.history = [st.session_state.history[-1]]  # Keep last
                st.rerun()

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Clustering Demo",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
        .main .block-container {padding-top: 1rem; padding-bottom: 1rem;}
        [data-testid="stMetricValue"] {font-size: 18px;}
        .stButton>button {border-radius: 5px; height: 45px;}
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    render_header()
    
    # Sidebar
    with st.sidebar:
        render_data_source()
        
        if st.session_state.data is not None:
            render_data_preview(st.session_state.data)
            st.divider()
            render_model_config()
            render_export()
    
    # Main content
    if st.session_state.data is not None:
        render_visualization()
        
        if st.session_state.get('trained'):
            st.divider()
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            metrics = st.session_state.metrics
            
            with col1:
                st.metric("Thời gian thực thi", f"{metrics['time']:.2f}s")
            
            with col2:
                st.metric("Silhouette Score", f"{metrics['silhouette']:.4f}")
                st.caption("Khoảng: [-1, 1]. Càng cao càng tốt.")
            
            with col3:
                st.caption("**Phân bố cụm:**")
                K = st.session_state.K
                labels = st.session_state.labels
                cluster_sizes = [np.sum(labels == k) for k in range(K)]
                
                # Bar chart
                fig_dist = go.Figure(data=[
                    go.Bar(x=[f'C{k+1}' for k in range(K)], y=cluster_sizes,
                          marker_color=['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', 
                                       '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#34495e'][:K])
                ])
                fig_dist.update_layout(
                    height=150,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis_title="Cụm",
                    yaxis_title="Số mẫu",
                    showlegend=False,
                    template='plotly_white'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
    else:
        st.info("Tải dữ liệu từ thanh bên để bắt đầu")
        
        st.markdown("""
        ### Hướng dẫn sử dụng:
        
        1. **Tải dữ liệu**: Tải lên file CSV với dữ liệu số (ví dụ: dữ liệu bán hàng, cảm biến, đặc trưng khách hàng)
        2. **Khám phá**: Xem trước dữ liệu ở thanh bên
        3. **Chọn K**: Chọn số cụm bằng thanh trượt (2-10)
        4. **Chọn phương pháp**: K-Means (nhanh) / Agglomerative (phân cấp) / Spectral (phi tuyến) / Ensemble (kết hợp)
        5. **Chạy**: Nhấn "Chạy Clustering" hoặc "Chạy tất cả" để phát hiện các nhóm
        6. **Phân tích**: Xem trực quan hóa và phân bố các cụm
        7. **Xuất**: Tải xuống kết quả với nhãn cụm
        
        ### Các phương pháp Clustering:
        
        - **K-Means**: Nhanh, dựa trên phân vùng, phù hợp với cụm hình cầu
        - **Agglomerative**: Phân cấp từ dưới lên, hiển thị mối quan hệ giữa các cụm
        - **Spectral**: Dựa trên đồ thị, xử lý được các mẫu phi tuyến phức tạp
        - **Ensemble**: Kết hợp cả 3 phương pháp để có kết quả ổn định nhất
        
        ### Ứng dụng:
        
        - Phân khúc khách hàng
        - Phân tích biểu hiện gen
        - Phân tích giỏ hàng
        - Phát hiện bất thường
        - Nén ảnh
        """)

if __name__ == "__main__":
    main()

