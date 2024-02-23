# Image Processing

Il codice Python fornito contiene una serie di funzioni per l'elaborazione delle immagini, tra cui filtri di sfocatura, sharpening, rilevamento dei bordi, trasformata di Fourier veloce (FFT), rimozione del rumore, equalizzazione dell'istogramma e altro ancora. Ecco un riassunto delle funzionalità principali:

1. **Blur e Blur Gaussian**: Queste funzioni applicano filtri di sfocatura all'immagine utilizzando rispettivamente un kernel box e un kernel gaussiano per ridurre la nitidezza e il rumore.

2. **Sharpening**: Questa funzione aumenta la nitidezza dell'immagine applicando ripetutamente una combinazione di filtri di sfocatura e differenza all'immagine.

3. **Sobel e Roberts Edge Detection**: Queste funzioni applicano rispettivamente il kernel di Sobel e il kernel di Roberts per rilevare i bordi dell'immagine.

4. **Fast Fourier Transform (FFT)**: Questa funzione calcola la trasformata di Fourier dell'immagine per analizzare le frequenze presenti.

5. **Median Filtering**: Questa funzione applica un filtro mediano all'immagine per rimuovere il rumore salt and pepper.

6. **Laplacian Filtering**: Questa funzione applica il kernel di Laplacian all'immagine per rilevare i dettagli di alta frequenza.

7. **Histogram Equalization**: Questa funzione equalizza l'istogramma dell'immagine per migliorare il contrasto e la luminosità.

8. **Salt and Pepper Noise**: Questa funzione aggiunge rumore "sale e pepe" all'immagine.

9. **Bit Slicing**: Questa funzione estrae i bit significativi dell'immagine per creare una nuova immagine con meno bit.

10. **Thresholding**: Esegue il thresholding dell'immagine binarizzandola in base a un valore di soglia `k`. I pixel con intensità inferiore a `k` vengono impostati a 0, mentre quelli con intensità maggiore o uguale a `k` vengono impostati a 255.
11. **Funzione logaritmo e esponenziale**: Applica una trasformazione logaritmica ed esponenziale all'immagine. Questa trasformazione può essere utilizzata per espandere (o diminuire nel caso dell'esponenziale) il range dinamico delle intensità dell'immagine, migliorando il contrasto nei dettagli più scuri (o più chiari). La funzione  permette la visualizzazione dell'immagine originale, dell'immagine trasformata con il logaritmo e dell'immagine trasformata con l'esponenziale. Vengono anche visualizzate le funzioni di trasformazione e gli istogrammi corrispondenti.

Il codice include anche una semplice interfaccia utente a riga di comando che consente all'utente di selezionare e applicare le diverse funzionalità al proprio file immagine di input. Tutte le funzionalità sono implementate utilizzando la libreria OpenCV e NumPy per l'elaborazione delle immagini e la libreria Matplotlib per la visualizzazione.
