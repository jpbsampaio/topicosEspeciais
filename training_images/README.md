# Imagens de Treinamento

Este diretório deve conter imagens para treinar o sistema de reconhecimento facial.

## Estrutura Recomendada:
```
training_images/
├── pessoa1/
│   ├── foto1.jpg
│   ├── foto2.jpg
│   └── foto3.jpg
├── pessoa2/
│   ├── foto1.jpg
│   └── foto2.jpg
└── ...
```

## Dicas para Melhores Resultados:
- Use imagens de boa qualidade (mínimo 300x300 pixels)
- Certifique-se de que o rosto está bem iluminado
- Evite óculos escuros ou chapéus que cubram o rosto
- Use múltiplas fotos de ângulos diferentes
- Uma única face por imagem funciona melhor

## Como Adicionar Faces:
1. Coloque as imagens no diretório training_images/
2. Use o cliente para adicionar faces conhecidas
3. Ou use a função add_face_from_file() no código
