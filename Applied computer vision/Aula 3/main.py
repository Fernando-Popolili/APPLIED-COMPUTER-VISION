from leitura_conversao import (
    load_image,
    convert_to_gray,
    show_image,
    plot_histogram,
    adjust_brightness_contrast,
    equalize_image,
    calculate_metrics,
    apply_threshold,
    apply_otsu,
    apply_morphology,
    find_contours,
    draw_contours,
    draw_bounding_boxes
)


if __name__ == "__main__":

    # --------------------------
    # 1. Carregar imagem
    # --------------------------
    img = load_image("sample.jpg")

    print("Shape imagem colorida:", img.shape)

    # --------------------------
    # 2. Converter para cinza
    # --------------------------
    gray = convert_to_gray(img)

    print("Shape imagem grayscale:", gray.shape)

    show_image(gray, "Imagem Original (Gray)")
    plot_histogram(gray)

    brightness, contrast = calculate_metrics(gray)

    print("\n--- MÉTRICAS ORIGINAIS ---")
    print("Brilho médio:", brightness)
    print("Contraste (desvio padrão):", contrast)

    # --------------------------
    # 3. Ajustar brilho e contraste
    # --------------------------
    adjusted = adjust_brightness_contrast(gray, alpha=1.3, beta=30)

    show_image(adjusted, "Brilho e Contraste Ajustados")
    plot_histogram(adjusted)

    brightness_adj, contrast_adj = calculate_metrics(adjusted)

    print("\n--- MÉTRICAS APÓS AJUSTE ---")
    print("Brilho médio:", brightness_adj)
    print("Contraste (desvio padrão):", contrast_adj)

    # --------------------------
    # 4. Equalizar histograma
    # --------------------------
    equalized = equalize_image(adjusted)

    show_image(equalized, "Imagem Equalizada")
    plot_histogram(equalized)

    brightness_eq, contrast_eq = calculate_metrics(equalized)

    print("\n--- MÉTRICAS APÓS EQUALIZAÇÃO ---")
    print("Brilho médio:", brightness_eq)
    print("Contraste (desvio padrão):", contrast_eq)

    # --------------------------
    # 5. Threshold manual
    # --------------------------
    binary_manual = apply_threshold(gray, threshold=120)
    show_image(binary_manual, "Threshold Manual")
    plot_histogram(binary_manual, "Histograma - Threshold Manual")

    # --------------------------
    # 6. Threshold automático (Otsu)
    # --------------------------
    binary_otsu = apply_otsu(gray)
    show_image(binary_otsu, "Threshold Otsu")
    plot_histogram(binary_otsu, "Histograma - Otsu")

    # --------------------------
    # 7. Morfologia para limpeza
    # --------------------------
    cleaned = apply_morphology(binary_otsu)
    show_image(cleaned, "Após Morfologia")

    # --------------------------
    # 8. Encontrar contornos
    # --------------------------
    contours = find_contours(cleaned)
    # Cada contorno representa um objeto detectado
    print("Quantidade de objetos detectados:", len(contours))

    # --------------------------
    # 9. Desenhar contornos
    # --------------------------
    img_contours = draw_contours(img, contours)
    show_image(img_contours, "Contornos Detectados")
    # --------------------------
    # 10. Bounding Boxes
    # --------------------------
    img_boxes = draw_bounding_boxes(img, contours)
    show_image(img_boxes, "Bounding Boxes")
'''
O que esse código mostra claramente:

Estrutura modular
Separação de responsabilidades
Transformações lineares
Impacto no histograma
Métricas quantitativas
Evolução incremental

Por que aparecem “lacunas” no histograma depois do ajuste?
Estamos fazendo uma transformação linear discreta sobre valores inteiros.

Imagens 8-bit têm apenas:

256 níveis possíveis (0–255)

São valores inteiros.

O que acontece na prática?

Suponha valores originais:
100, 101, 102, 103

Se você aplicar:
alpha = 1.5
beta = 0

Cálculo:
100 → 150
101 → 151.5 → 152
102 → 153
103 → 154.5 → 155

Perceba algo importante:
Alguns valores “pulam”
Nem todos os números entre 150 e 155 aparecem
Isso gera buracos no histograma.

O histograma mostra que imagem digital é um sistema discreto.
Quando esticamos os valores, criamos espaços vazios porque não existem infinitos níveis
'''

    