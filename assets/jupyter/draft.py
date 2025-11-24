# utils
def generate_3d(data, noise, samples, shuffle=False):
    if data == 1:
        centers = [(1, 0, 0), 
                   (0, 1, 0), 
                   (-1, 0, 0)]
    elif data == 2:
        centers = [(np.cos(np.pi/4), np.sin(np.pi/4), 1),
                   (np.cos(2*np.pi/3), np.sin(2*np.pi/3), 1),
                   (np.cos(np.pi), np.sin(np.pi), 1)]
    elif data == 3:
        centers = [(np.cos(np.pi/4), np.sin(np.pi/4), 1),
                   (np.cos(2*np.pi/3), np.sin(2*np.pi/3), 1),
                   (np.cos(5*np.pi/6), np.cos(5*np.pi/6), 1)]
    else:
        raise NameError('Data not found.')

    X, Y = [], []
    for c, center in enumerate(centers):
        _X = np.random.normal(center, scale=(noise, noise, noise), size=(samples, 3))
        _Y = np.ones(samples, dtype=np.int32) * c
        X.append(_X)
        Y.append(_Y)
    X = np.vstack(X)
    X = X / np.linalg.norm(X, axis=1, ord=2, keepdims=True)
    Y = np.hstack(Y)
    
    if shuffle:
        idx_arr = np.random.choice(np.arange(len(X)), len(X), replace=False)
        X, Y = X[idx_arr], Y[idx_arr]

    X = torch.tensor(X).float()
    Y = torch.tensor(Y).long()
    return X, Y, 3


def plot_3d(Z, y, title=''):
    import plotly.graph_objects as go
    
    # Convert to numpy if tensor
    if hasattr(Z, 'numpy'):
        Z = Z.numpy()
    if hasattr(y, 'numpy'):
        y = y.numpy()
    
    colors = ['forestgreen', 'royalblue', 'brown']
    
    fig = go.Figure()
    
    # Add scatter points for each cluster
    for c in np.unique(y):
        mask = y == c
        fig.add_trace(go.Scatter3d(
            x=Z[mask, 0],
            y=Z[mask, 1],
            z=Z[mask, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors[c],
                opacity=0.8
            ),
            name=f'Cluster {c}',
            showlegend=True
        ))
    
    # Add wireframe sphere
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(go.Surface(
        x=x_sphere,
        y=y_sphere,
        z=z_sphere,
        colorscale=[[0, 'rgba(128,128,128,0.5)'], [1, 'rgba(128,128,128,0.5)']],
        showscale=False,
        opacity=0.3,
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            zaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=700,
        height=500,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.show()