# Image Generation with GAN

This project explores **image generation using Generative Adversarial Networks (GANs)**. Different configurations of **latent dimension (`latent_dim`)** and **hidden dimension (`hidden_dim`)** were tested to analyze their impact on generation quality, as measured by **FID scores** and interpolation results.

---

## 📦 Environment

- **Python**: 3.8  
- **PyTorch**: 1.1  
- **torchvision**  
- **TensorBoard**  
- **SciPy**: 1.3  
- **Hardware**: MacBook Pro (Apple M1 chip)  

---

## ⚙️ Parameters

- **Batch size**: 64  
- **Training steps**: 5000  
- **Latent dimension (`latent_dim`)**: [16, 100]  
- **Hidden dimension (`hidden_dim`)**: [16, 100]  

---

## 📊 Results

### FID Scores

| latent_dim | hidden_dim | FID Score |
|------------|------------|-----------|
| 16         | 16         | 83.66 |
| 100        | 16         | 79.20 |
| 16         | 100        | 43.37 |
| 100        | 100        | **41.94** |

👉 Best performance achieved with **latent_dim = 100** and **hidden_dim = 100**, showing the lowest FID score:contentReference[oaicite:0]{index=0}.  

---

## 🔍 Analysis

### Impact of `hidden_dim`
- Larger `hidden_dim` increases model capacity, allowing the generator to learn **finer details**.  
- However, too large values can risk **overfitting** and unstable training.  
- Increasing from 16 → 100 improved FID significantly.  

### Impact of `latent_dim`
- Larger latent spaces provide **more variability**, improving diversity of generated images.  
- Increasing latent_dim from 16 → 100 improved FID, especially when paired with a larger hidden_dim.  

### Combined Effect
- **Both latent_dim and hidden_dim need to scale together** for best results.  
- The **latent_dim = 100, hidden_dim = 100** configuration achieved the best quality images.  

---

## ⚖️ Nash Equilibrium

- GAN training did **not converge cleanly to Nash Equilibrium**.  
- Typically, the **Discriminator learns faster** initially, forcing the Generator to catch up.  
- Over time, the Generator improves, narrowing the gap, but training remains an oscillatory process:contentReference[oaicite:1]{index=1}.  

---

## 🔄 Latent Space Interpolation

Interpolation experiments show smooth transitions between digits, confirming that the generator:  
- Navigates latent space effectively.  
- Produces realistic, gradual transformations without abrupt artifacts.  
- Avoids **mode collapse**, maintaining diversity:contentReference[oaicite:2]{index=2}.  

Example interpolation results:  

![Interpolated Samples](docs/interpolation.png)

---

## ✅ Key Takeaways

1. **Larger latent and hidden dimensions improve generative quality**, but must be balanced.  
2. **Best results**: latent_dim = 100, hidden_dim = 100.  
3. **Training remains adversarial and oscillatory**, without full Nash Equilibrium.  
4. **Interpolation confirms generator’s ability** to produce diverse and realistic images.  
