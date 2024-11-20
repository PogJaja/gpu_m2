import numpy as np

def scanCPU(array):
    # Vérifier que la taille du tableau est une puissance de 2
    n = len(array)
    if (n & (n - 1)) != 0:
        raise ValueError("La taille du tableau doit être une puissance de 2.")
    m = np.log2(n).astype(int)

    # Phase de montée
    for d in range (0, m): # Parcours de 0 à m - 1 inclus
        for k in range (0, n, 2 ** (d + 1)): # Parcours de 0 à n - 1 inclus
            array[k + 2 ** (d+1) - 1] += array[k + 2 ** d - 1]
            print (array)

    # Mettre le dernier élément à zéro
    array[n - 1] = 0
    print (array)

    # Phase de descente
    for d in range (m - 1, -1, -1): # Parcours de m- 1 à 0 inclus
        for k in range (0, n, 2 ** (d + 1)): # Parcours de 0 à n - 1 inclus
            temp = array[k + 2 ** d - 1]
            array[k + 2 ** d - 1] = array[k + 2 ** (d+1) - 1]
            array[k + 2 ** (d+1) - 1] += temp
            print (array)

    return array

#Test
if __name__ == "__main__":

    n = 4  # Taille (doit être une puissance de 2)
    array = [2, 3, 4, 6]
    print("Tableau initial:", array)

    result = scanCPU(array)
    print("Tableau final (prefix exclusif):", result)
