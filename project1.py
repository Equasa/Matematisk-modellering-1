import numpy as np
import matplotlib.pyplot as plt
import pulp

np.random.seed(42)

factories = np.array([[10, 260], [130, 120], [50, 70]])  # Fabrikernas koordinater
produktionsförmåga = [400, 200, 300]
grossister = np.vstack((np.random.randint(0, 361, 8), np.random.randint(0, 326, 8))).transpose()
efterfrågan = np.random.randint(50, 151, len(grossister))

def avstånd(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def lp(factories, produktionsförmåga, grossister, efterfrågan):
    transport_kostnader = np.array([[avstånd(f, g) for g in grossister] for f in factories])

    # Skapa ett linjärprogrammeringsproblem med PuLP för att minimera den totala transportkostnaden
    prob = pulp.LpProblem("Minimera_Transportavstånd", pulp.LpMinimize)

    # Skapa beslutsvariabler
    transport_var = pulp.LpVariable.dicts("Transport",
                                        ((i, j) for i in range(len(factories)) for j in range(len(grossister))),
                                        lowBound=0,
                                        cat='Continuous')

    # Målfunktion: Minimera totala transportkostnaden (avståndet)
    prob += pulp.lpSum([transport_kostnader[i][j] * transport_var[(i, j)] 
                        for i in range(len(factories)) for j in range(len(grossister))])

    # Begränsningar
    for j in range(len(grossister)):
        prob += pulp.lpSum([transport_var[(i, j)] for i in range(len(factories))]) == efterfrågan[j], f"Efterfrågan_{j}"

    for i in range(len(factories)):
        prob += pulp.lpSum([transport_var[(i, j)] for j in range(len(grossister))]) <= produktionsförmåga[i], f"Kapacitet_{i}"

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    objective_value = pulp.value(prob.objective)

    transport_quantities = np.array([[transport_var[(i, j)].varValue for j in range(len(grossister))] for i in range(len(factories))])

    return objective_value, transport_quantities


original_obj_val, original_transports = lp(factories, produktionsförmåga, grossister, efterfrågan)
print(f"Original distance: {original_obj_val}")
print(f"Original transports: \n{original_transports}")


min_total_distance = np.inf
optimal_location = None
resolution = 2
for x in range(0, 361, resolution):
    for y in range(0, 326, resolution):
        new_factory_location = np.array([x, y])
        factories_with_new = np.vstack((factories, new_factory_location))

        produktionsförmåga_with_new = np.append(produktionsförmåga, 900)

        current_total_distance, transports = lp(factories_with_new, produktionsförmåga_with_new, grossister, efterfrågan)

        if current_total_distance < min_total_distance:
            min_factories = factories_with_new
            min_transports = transports
            min_total_distance = current_total_distance
            optimal_location = new_factory_location


print(f"Min {min_total_distance}")
print(f"Optimal location for 4th factory: {optimal_location}")
print(min_transports)


plt.figure(figsize=(12, 8))

for i, (x, y) in enumerate(factories):
    plt.scatter(x, y, color='red', s=100)
    plt.text(x, y, f'Factory {i+1}', horizontalalignment='right')

for i, (x, y) in enumerate(grossister):
    plt.scatter(x, y, color='blue', s=50)
    plt.text(x, y, f'Grossist {i+1}', horizontalalignment='right')

plt.scatter(optimal_location[0], optimal_location[1], color='green', s=100)
plt.text(optimal_location[0], optimal_location[1], 'New Factory', horizontalalignment='right')

for i, factory_transports in enumerate(min_transports):
    for j, quantity in enumerate(factory_transports):
        if quantity > 0:  
            plt.plot([min_factories[i][0], grossister[j][0]], [min_factories[i][1], grossister[j][1]], 'k--', linewidth=0.5)

plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Factory and Grossist Transports with New Optimal Factory Location')
plt.grid(True)
plt.show()