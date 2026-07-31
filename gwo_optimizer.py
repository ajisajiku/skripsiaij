import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import random
import streamlit as st

def calculate_fitness(params, X_train_scaled, y_train):
    max_depth = int(params[0])
    min_samples_leaf = int(params[1])

    if max_depth < 1 or min_samples_leaf < 1:
        return 1.0

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        criterion='gini',
        random_state=42
    )

    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='recall')
    avg_recall = np.mean(scores)

    return 1.0 - avg_recall

def run_gwo(fitness_function, lower_bound, upper_bound, dimensions, num_wolves, max_iterations, X_train_scaled, y_train):
    lower_bound = np.array(lower_bound)
    upper_bound = np.array(upper_bound)

    alpha_pos = np.zeros(dimensions)
    alpha_score = float('inf')
    beta_pos = np.zeros(dimensions)
    beta_score = float('inf')
    delta_pos = np.zeros(dimensions)
    delta_score = float('inf')

    positions = np.zeros((num_wolves, dimensions))
    for i in range(num_wolves):
        positions[i, :] = np.random.uniform(0, 1, dimensions) * (upper_bound - lower_bound) + lower_bound

    progress_bar = st.progress(0)
    status_text = st.empty()

    for t in range(max_iterations):
        for i in range(num_wolves):
            positions[i,:] = np.clip(positions[i,:], lower_bound, upper_bound)
            fitness = fitness_function(positions[i, :], X_train_scaled, y_train)

            if fitness < alpha_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = alpha_score
                beta_pos = alpha_pos.copy()
                alpha_score = fitness
                alpha_pos = positions[i, :].copy()
            elif fitness > alpha_score and fitness < beta_score:
                delta_score = beta_score
                delta_pos = beta_pos.copy()
                beta_score = fitness
                beta_pos = positions[i, :].copy()
            elif fitness > alpha_score and fitness > beta_score and fitness < delta_score:
                delta_score = fitness
                delta_pos = positions[i, :].copy()

        yield {
           'iteration': t + 1,
           'alpha_params': alpha_pos,
           'alpha_score': alpha_score,
           'beta_params': beta_pos,
           'beta_score': beta_score,
           'delta_params': delta_pos,
           'delta_score': delta_score
        }

        a = 2 - t * (2 / max_iterations)

        for i in range(num_wolves):
            for j in range(dimensions):
                r1, r2 = random.random(), random.random()
                A1 = 2 * a * r1 - a
                C1 = 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = random.random(), random.random()
                A2 = 2 * a * r1 - a
                C2 = 2 * r2
                D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = random.random(), random.random()
                A3 = 2 * a * r1 - a
                C3 = 2 * r2
                D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                positions[i, j] = (X1 + X2 + X3) / 3
        
        progress_bar.progress((t + 1) / max_iterations)
        status_text.text(f'Iterasi {t+1}/{max_iterations}, Best Score (1 - Recall): {alpha_score:.4f}')
