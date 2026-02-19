import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
def precision_at_k(y_true, y_scores, k):
    """
    y_true: array-like (0/1)
    y_scores: predicted probabilities
    k: top K candidates
    """
    # сортуємо по скору
    order = np.argsort(y_scores)[::-1]
    top_k = order[:k]

    return y_true[top_k].mean()


def recall_at_k(y_true, y_scores, k):
    """
    Частка всіх Senior, які потрапили в top-K
    """
    order = np.argsort(y_scores)[::-1]
    top_k = order[:k]

    return y_true[top_k].sum() / y_true.sum()

def evaluate_test_accuracy(y_true, y_pred, dataset_name="Test"):
    """
    Спрощена функція тільки для accuracy
    
    Parameters:
    -----------
    y_true : array-like
        Справжні значення (0/1)
    y_pred : array-like
        Прогнозовані значення (0/1)
    dataset_name : str
        Назва датасету (Test/Train)
    
    Returns:
    --------
    float: accuracy score
    """
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✅ {dataset_name} Accuracy: {accuracy:.3f}")
    return accuracy
def pr_treshold_curve(y_proba,y_test):
    from sklearn.metrics import precision_recall_curve


    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    for p, r, t in zip(precision[:5], recall[:5], thresholds[:5]):
        print(f"Threshold={t:.2f} | Precision={p:.2f} | Recall={r:.2f}")
def plot_calibration_simple(y_true, y_prob):
    """
    Найпростіша калібрувальна крива
    y_true: справжні лейбли (0/1)
    y_prob: ймовірності від моделі
    """
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt
    
    # Обчислюємо калібрувальну криву
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
    
    # Малюємо графік
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 's-', label='Модель')
    plt.plot([0, 1], [0, 1], 'k--', label='Ідеальна калібрування')
    plt.xlabel('Середня прогнозована ймовірність')
    plt.ylabel('Частка позитивних')
    plt.title('Калібрувальна крива')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Виводимо прості результати
    print(f"Біни: {prob_pred}")
    print(f"Реальні частки: {prob_true}")