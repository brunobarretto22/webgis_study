# streamlit run .\main.py

import streamlit as st
import geopandas as gpd
import pandas as pd
from zona_utm import calcular_utm
import folium
from streamlit_folium import st_folium
from folium.raster_layers import ImageOverlay
from rasterio.io import MemoryFile
from rasterio.mask import mask
import numpy as np
from utils import color_map,value_to_class
import cv2
# pip install ploty-express
import plotly.express as px

st.header('Study Webgis')

st.sidebar.title('Menu')

poligono_upload = st.sidebar.file_uploader('Escolha o polígono:')

raster_upload = st.sidebar.file_uploader('Escolha o raster a ser utilizado na análise (Mapbiomas):')

embargos_ibama = 'adm_embargos_ibama_a_mt.parquet''
municipios_mt = 'BR_Municipios_2024_b0_mt.shp'
uso_consolidado = 'USO_CONSOLIDADO_b0_porto_esperidiao.shp'

# adicionar depois dados do IBGE e autos de infração

poligono_analise = gpd.read_file(poligono_upload)  
# st.write(poligono_analise)

if poligono_upload:
    poligono_analise = gpd.read_file(poligono_upload)

    @st.cache_resource
    def abrir_embargo():
        gdf_embargo = gpd.read_parquet(embargos_ibama)
        return gdf_embargo
    
    gdf_embargo = abrir_embargo()
    
    @st.cache_resource
    def abrir_municipios_mt():
        gdf_municipios_mt = gpd.read_file(municipios_mt)
        return gdf_municipios_mt
    
    gdf_municipios_mt = abrir_municipios_mt()

    @st.cache_resource
    def abrir_uso_consolidado():
        gdf_uso_consolidado = gpd.read_file(uso_consolidado)
        return gdf_uso_consolidado
    
    gdf_uso_consolidado = abrir_uso_consolidado()

    # st.write(gdf_embargo)
    # Spation join como? somente da parte que se sobrepõe
    entrada_embargo = gpd.sjoin(gdf_embargo, poligono_analise, how='inner', predicate='intersects')
    # linha que vai fazer o cruzamento
    entrada_embargo = gpd.overlay(entrada_embargo, poligono_analise, how='intersection')
    # Spation join como? somente da parte que se sobrepõe
    entrada_municipios_mt = gpd.sjoin(gdf_municipios_mt, poligono_analise, how='inner', predicate='intersects')
    # linha que vai fazer o cruzamento
    entrada_municipios_mt = gpd.overlay(entrada_municipios_mt, poligono_analise, how='intersection')
    # Spation join como? somente da parte que se sobrepõe
    entrada_uso_consolidado = gpd.sjoin(gdf_uso_consolidado, poligono_analise, how='inner', predicate='intersects')
    # linha que vai fazer o cruzamento
    entrada_uso_consolidado = gpd.overlay(entrada_uso_consolidado, poligono_analise, how='intersection')

    epsg_arquivo = calcular_utm(poligono_analise)
    
    area_embargo = entrada_embargo.dissolve(by=None)

    area_embargo = area_embargo.to_crs(epsg=epsg_arquivo)
    
    area_municipios_mt = entrada_municipios_mt.dissolve(by=None)

    area_municipios_mt = area_municipios_mt.to_crs(epsg=epsg_arquivo)

    area_uso_consolidado = entrada_uso_consolidado.dissolve(by=None)

    area_uso_consolidado = area_uso_consolidado.to_crs(epsg=epsg_arquivo)
    
    # st.write(epsg_arquivo)

    with MemoryFile(raster_upload.getvalue()) as memfile:
            with memfile.open() as src:

                if poligono_analise.crs != src.crs:
                    poligono_analise = poligono_analise.to_crs(src.crs)
                
                geometries = poligono_analise.geometry
                out_image, out_transform = mask(src, geometries, crop=True)
                out_image = out_image[0]

                height, width = out_image.shape

                rgb_image = np.zeros((height,width,4),dtype=np.uint8)

                for value,color in color_map.items():
                    rgb_image[out_image == value] = color

                resized_image = cv2.resize(rgb_image,(width,height),interpolation=cv2.INTER_NEAREST)

                min_x,min_y = out_transform * (0,0)
                max_x,max_y = out_transform * (width,height)

                bounds = [[min_y,min_x], [max_y,max_x]]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Área embargada (ha)')
        if len(area_embargo) == 0:
            st.subheader('0.0000')
        else:
            area_embargo['area'] = area_embargo.area / 10000
            st.subheader(str(round(area_embargo.loc[0, 'area'], 4)))
    with col2:
        st.subheader('Área Município MT (ha)')
        if len(area_municipios_mt) == 0:
            st.subheader('0.0000')
        else:
            area_municipios_mt['area'] = area_municipios_mt.area / 10000
            st.subheader(str(round(area_municipios_mt.loc[0, 'area'], 4)))
    with col3:
        st.subheader('Área Uso consolidado (ha)')
        if len(area_uso_consolidado) == 0:
            st.subheader('0.0000')
        else:
            area_uso_consolidado['area'] = area_uso_consolidado.area / 10000
            st.subheader(str(round(area_uso_consolidado.loc[0, 'area'], 4)))

    centroid_x,centroid_y = poligono_analise.centroid.x, poligono_analise.centroid.y;

    m = folium.Map(location=[centroid_y,centroid_x],zoom_start=8,tiles='Esri.WorldImagery')

    ImageOverlay(
    image=resized_image,
    bounds=bounds,
    opacity=0.7,
    name='Mapbiomas coleção 9',
    interactive=True,
    cross_origin=False,
    zindex=1
    ).add_to(m)

    minx,miny,maxx,maxy = poligono_analise.total_bounds

    bounds = [[miny,minx],[maxy,maxx]]

    m.fit_bounds(bounds)

    def style_function_poligono_analise(x): return{
        'fillColor':'white',
        'color':'black',
        'weight':1,
        'fillOpacity':0.6
    }

    poligono_analise_geom = gpd.GeoDataFrame(poligono_analise,columns=['geometry'])
    folium.GeoJson(poligono_analise_geom,name='Polígono em análise').add_to(m)
    def style_function_embargos_ibama(x): return{
        'fillColor':'green',
        'color':'black',
        'weight':1,
        'fillOpacity':0.6
    }
    
    # embargos_ibama_geom = gpd.GeoDataFrame(embargos_ibama,columns=['geometry'])
    # folium.GeoJson(embargos_ibama_geom,name='Embargos IBAMA').add_to(m)
    def style_function_municipios_mt(x): return{
        'fillColor':'yellow',
        'color':'black',
        'weight':1,
        'fillOpacity':0.6
    }
    # municipios_mt_geom = gpd.GeoDataFrame(municipios_mt,columns=['geometry'])
    # folium.GeoJson(municipios_mt_geom,name='Municípios-MT').add_to(m)

    def style_function_uso_consolidado(x): return{
        'fillColor':'red',
        'color':'black',
        'weight':1,
        'fillOpacity':0.6
    }
    # uso_consolidado_geom = gpd.GeoDataFrame(uso_consolidado,columns=['geometry'])
    # folium.GeoJson(uso_consolidado_geom,name='Uso consolidado SEMA-MT').add_to(m)

    folium.LayerControl().add_to(m)

    st_folium(m,width='100%')

    unique_values,counts = np.unique(out_image,return_counts=True)
    st.write('Áreas em hectares')
    for value,count in zip (unique_values,counts):
            class_name = value_to_class.get(value, "Unknown")
            area_ha = (count * 900) / 10000
            st.write(f'{class_name},{area_ha} (ha)')
    

    # Gráfico Interativo aula 2.6
    df_embargo = pd.DataFrame(entrada_embargo).drop(columns=['geometry'])

    df_municipios_mt = pd.DataFrame(entrada_municipios_mt).drop(columns=['geometry'])

    df_uso_consolidado = pd.DataFrame(entrada_uso_consolidado).drop(columns=['geometry'])

    col1_graf, col2_graf, col3_graf, col4_graf = st.columns(4)

    tema_grafico = col1_graf.selectbox('Selecione o tema do gráfico',options=['Embargo', 'Municípios-MT', 'Uso consolidado'])

    if tema_grafico == 'Embargo':
        df_analisado = df_embargo
    elif tema_grafico == 'Municípios-MT':
        df_analisado = df_municipios_mt
    elif tema_grafico == 'Uso consolidado':
        df_analisado = df_uso_consolidado;

    tipo_grafico = col2_graf.selectbox('Selecione o tipo de gráfico',
                                       options=['box','bar','line','scatter','violin','histogram'],index=5)
    
    plot_func = getattr(px,tipo_grafico)

    x_val = col3_graf.selectbox('Selecione o eixo x do gráfico',options=df_analisado.columns,index=6)

    y_val = col4_graf.selectbox('Selecione o eixo y do gráfico',options=df_analisado.columns,index=6)

    plot = plot_func(df_analisado,x=x_val,y=y_val)


    st.plotly_chart(plot,use_container_width=True);


