import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC


# 1. CARREGAMENTO DO DATASET

data = load_breast_cancer()
X = data.data
y = data.target

print("Dataset carregado com sucesso!")


# 2. GRID SEARCH (Busca em Grade)


print("\nIniciando Grid Search...")

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'gamma': [0.0001, 0.001, 0.01, 0.1, 1, 10]
}

svm = SVC(kernel='rbf')

grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1
)

start_time_grid = time.time()
grid_search.fit(X, y)
end_time_grid = time.time()

time_grid = end_time_grid - start_time_grid
best_params_grid = grid_search.best_params_
best_acc_grid = grid_search.best_score_

print(f"\nMelhor acurácia (Grid): {best_acc_grid:.4f}")
print(f"Melhores parâmetros (Grid): {best_params_grid}")
print(f"Tempo de execução (Grid): {time_grid:.4f} segundos")

import pandas as pd
results = pd.DataFrame(grid_search.cv_results_)

# Selecionar apenas colunas relevantes
tabela = results[[
    'param_C',
    'param_gamma',
    'mean_test_score',
    'std_test_score',
    'rank_test_score'
]]

# Renomear colunas
tabela = tabela.rename(columns={
    'param_C': 'C',
    'param_gamma': 'Gamma',
    'mean_test_score': 'Acurácia Média',
    'std_test_score': 'Desvio Padrão',
    'rank_test_score': 'Ranking'
})

print ("\n---- Resuldados complestos do Grid Search ----\n")
print(tabela)

#Gerar o grafico da tabela
plt.figure(figsize=(20, 4))
plt.title("Tabela de Resultados do Grid Search", fontsize=14, pad=20)

plt.table(
    cellText=tabela.values,
    colLabels=tabela.columns,
    loc='center',
    cellLoc='center'
)
plt.axis('off')
plt.show()



# 3. IMPLEMENTAÇÃO DO PSO


print("\nIniciando PSO...")

def fitness_function(position):
    c_val = position[0]
    g_val = position[1]

    if c_val <= 0 or g_val <= 0:
        return 0.0

    clf = SVC(kernel='rbf', C=c_val, gamma=g_val)
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
    return scores.mean()


num_particles = 20
num_iterations = 100
w = 0.5
c1 = 1.5
c2 = 1.5

bounds_min = np.array([0.01, 0.0001])
bounds_max = np.array([1000.0, 10.0])

particles_position = np.random.uniform(bounds_min, bounds_max, (num_particles, 2))
particles_velocity = np.random.uniform(-1, 1, (num_particles, 2))

pbest_position = particles_position.copy()
pbest_fitness = np.array([0.0] * num_particles)
gbest_position = np.array([0.0, 0.0])
gbest_fitness = 0.0

history_fitness = []

start_time_pso = time.time()

for i in range(num_iterations):
    for j in range(num_particles):
        fitness = fitness_function(particles_position[j])

        if fitness > pbest_fitness[j]:
            pbest_fitness[j] = fitness
            pbest_position[j] = particles_position[j].copy()

        if fitness > gbest_fitness:
            gbest_fitness = fitness
            gbest_position = particles_position[j].copy()

    for j in range(num_particles):
        r1 = random.random()
        r2 = random.random()

        new_velocity = (w * particles_velocity[j]) + \
                       (c1 * r1 * (pbest_position[j] - particles_position[j])) + \
                       (c2 * r2 * (gbest_position - particles_position[j]))

        particles_velocity[j] = new_velocity
        particles_position[j] = particles_position[j] + new_velocity

        particles_position[j] = np.clip(particles_position[j], bounds_min, bounds_max)

    history_fitness.append(gbest_fitness)

    if i % 10 == 0:
        print(f"Iteração {i}/{num_iterations} - Melhor Acurácia: {gbest_fitness:.4f}")

end_time_pso = time.time()
time_pso = end_time_pso - start_time_pso

print(f"\nMelhor acurácia (PSO): {gbest_fitness:.4f}")
print(f"Melhores parâmetros (PSO): C={gbest_position[0]:.4f}, Gamma={gbest_position[1]:.4f}")
print(f"Tempo de execução (PSO): {time_pso:.4f} segundos")



# 4. GRÁFICO — Evolução do PSO

plt.figure(figsize=(10, 5))
plt.plot(history_fitness, label="Melhor Fitness (PSO)", linewidth=2)
plt.xlabel("Iteração")
plt.ylabel("Acurácia")
plt.title("Evolução da Acurácia ao Longo das Iterações - PSO")
plt.grid(True)
plt.legend()
plt.show()



# 5. GRÁFICO — Grid Search vs PSO

metodos = ["Grid Search", "PSO"]
acuracias = [best_acc_grid, gbest_fitness]

plt.figure(figsize=(7, 5))
sns.barplot(x=metodos, y=acuracias)
plt.ylabel("Acurácia Média")
plt.ylim(0.0, 1.0)
plt.title("Comparação: Grid Search vs PSO")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()


# 6. HEATMAP DO GRID SEARCH

results = grid_search.cv_results_
C_vals = param_grid['C']
G_vals = param_grid['gamma']

heatmap_matrix = results['mean_test_score'].reshape(len(C_vals), len(G_vals))

plt.figure(figsize=(10, 7))
sns.heatmap(
    heatmap_matrix,
    annot=True,
    xticklabels=G_vals,
    yticklabels=C_vals,
    cmap="viridis"
)
plt.title("Heatmap do Grid Search (Acurácia)")
plt.xlabel("Gamma")
plt.ylabel("C")
plt.show()


# 7. GRÁFICO em 3d do Grid Search

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

C_mesh, G_mesh = np.meshgrid(C_vals, G_vals)
Z = heatmap_matrix.T

ax.plot_surface(C_mesh, G_mesh, Z, cmap="viridis")
ax.set_xlabel("C")
ax.set_ylabel("Gamma")
ax.set_zlabel("Acurácia")
ax.set_title("Superfície 3D — Grid Search")
plt.show()
