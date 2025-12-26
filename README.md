# Ai-final-project

# 🤖 AI From Scratch: Linear & Logistic Regression Project

Bu layihə süni zəka dərsi üçün hazırlanmışdır. Layihənin əsas məqsədi, hazır kitabxanalardan (məsələn, Scikit-Learn) istifadə etmədən, riyazi alqoritmləri **sıfırdan (from scratch)** NumPy vasitəsilə tətbiq etməkdir.

## 🚀 Layihənin Xüsusiyyətləri
Bu layihə iki əsas mərhələdən ibarətdir:
1.  **Linear Regression:** Ev qiymətlərinin proqnozu (Kəmiyyət analizi).
2.  **Logistic Regression:** Döş xərçənginin diaqnozu (Bədxassəli/Xoşxassəli təsnifatı).

### 🧠 Riyazi Nüvə (From Scratch)
Modelin daxilində tətbiq etdiyimiz əsas riyazi komponentlər:
* **Sigmoid Function:** Xətti nəticəni ehtimala çevirmək üçün.
    $$g(z) = \frac{1}{1 + e^{-z}}$$
* **Log Loss (Binary Cross-Entropy):** Təsnifat xətasını minimuma endirmək üçün.
* **Gradient Descent:** Ən yaxşı $\theta$ (çəki) parametrlərini tapmaq üçün istifadə olunan optimallaşdırma alqoritmi.

## 💻 Necə İşlətməli?

1.  **Kitabxanaları yükləyin:**
    ```bash
    pip install streamlit numpy pandas scikit-learn matplotlib
    ```

2.  **Tətbiqi başladın:**
    ```bash
    streamlit run app.py
    ```

## 📊 Nəticələr
Bizim sıfırdan yazdığımız modelin nəticələri Scikit-Learn kitabxanası ilə müqayisə edilmiş və yüksək dəqiqlik (Accuracy) əldə olunmuşdur. Logistik reqressiya modeli tibbi datalar üzərində uğurla sınaqdan keçmişdir.

---
*Bu layihə AI kursunun final işi olaraq hazırlanmışdır.*
