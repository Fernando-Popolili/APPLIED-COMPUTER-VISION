# este é o arquivo "principal" da aplicação - entry point

#para executar as funções definidas no outro arquivo, precisamos importar os recursos

from leitura_conversao import(
    carregarImagem,
    converterCinza,
    mostrarImagem,
    histograma,
    ajustarBrilhoContraste,
    equalizar_histograma,
    calcular_metricas
)


if __name__ == '__main__':

#Carregar a imagem
    img = carregarImagem('imagem.jpg')
    print("Shape Imagem colorida", img.shape)
    
    gray = converterCinza(img)
    print("shape imagem escala de cinza", gray.shape)

    mostrarImagem(gray, "Imagem original(Gray)")
    histograma(gray)

    brilho, contraste = calcular_metricas(gray)

    print("----------------Métricas originais----------------")
    print("Brilho médio", brilho)
    print("contraste (obtdo a partir do desvio padrao):", contraste)

#3. Ajuste de brilho e contraste
    ajuste = ajustarBrilhoContraste(gray, alpha=1.3, beta=30)
    mostrarImagem(ajuste, "Brilho e Contraste ajustados")
    histograma(ajuste)


    print("----------------Métricas após ajustes----------------")
    print("Brilho médio", brilho)
    print("contraste (obtdo a partir do desvio padrao):", contraste)

#4. Equalizando o histograma
    equalizar = equalizar_histograma(ajuste)
    mostrarImagem(equalizar, "Imagem equalizada a partir do brilho e constraste")
    histograma(equalizar)



    brilho_eq, contraste_eq = calcular_metricas(equalizar)
    print("----------------Métricas após equalização----------------")
    print("Brilho médio", brilho_eq)
    print("contraste (obtdo a partir do desvio padrao):", contraste_eq)