function draw(data,y_segmentation, y_axis_type, svg){

          "use strict";
          d3.select("svg").remove()

          var margin = 375,
              width = 1400 - margin,
              height = 600 - margin;

          var svg = d3.select("body")
            .append("svg")
              .attr("width", width + margin)
              .attr("height", height + margin)
            .append('g')
                .attr('class','chart');

          var myChart = new dimple.chart(svg, data);
          var x = myChart.addTimeAxis("x","NewLoanOriginationDate","%m-%Y","%m-%Y"); 


          if(y_axis_type == "Percentage"){
            var y = myChart.addPctAxis("y","LoanOriginalAmount");
            myChart.addSeries(y_segmentation, dimple.plot.bar);
          }
          else if(y_axis_type == "Absolute"){
            var y = myChart.addMeasureAxis("y","LoanOriginalAmount");
            myChart.addSeries(y_segmentation, dimple.plot.area);
          }
              

          myChart.addLegend(60, 10, 500, 20, "right");
          myChart.draw();
};


function first_draw(data) {
      
    	/*
        D3.js setup code
     	 */

          "use strict";
          var margin = 375,
              width = 1400 - margin,
              height = 600 - margin;

          d3.select("body")
            .append("h2")
            .text("The 2008 Subprime crisis - Loan Data from Prosper")

          var svg = d3.select("body")
            .append("svg")
              .attr("width", width + margin)
              .attr("height", height + margin)
            .append('g')
                .attr('class','chart');


          var categories = ["IncomeRange","IncomeVerifiable","LoanStatus","IsBorrowerHomeowner"];

          var buttons = d3.select("body")
          .append("div")
          .attr("class","segmentation_buttons")
          .selectAll("div")
          .data(categories)
          .enter()
          .append("div")
          .text(function(d){
          	return d;
          });

          var graph_styles = ["Percentage","Absolute"]
          var graph_selector = d3.select("body")
          .append("div")
          .attr("class","graph_styles")
          .selectAll("div")
          .data(graph_styles)
          .enter()
          .append("div")
          .text(function(d){
            return d;
          });

          var last_button;
          var last_style;

          buttons.on("click",function(d){
          	d3.select(".segmentation_buttons")
          	.selectAll("div")
          	.transition()
          	.duration(500)
          	.style("color","black")
          	.style("background","rgb(251,201,127)")

          	d3.select(this)
          	.transition()
          	.duration(500)
          	.style("background","lightBlue")
          	.style("color","white")

            last_button = d3.select(this).text();

            draw(data,last_button,last_style,svg);
          })

          buttons.on("mouseover",function(d){
            d3.select(".segmentation_buttons")
            .style("cursor","pointer")
          })

          graph_selector.on("click",function(d){
            d3.select(".graph_styles")
            .selectAll("div")
            .transition()
            .duration(500)
            .style("color","black")
            .style("background","white")

            d3.select(this)
            .transition()
            .duration(500)
            .style("background","black")
            .style("color","white")

            last_style = d3.select(this).text();

            draw(data,last_button,last_style,svg);

          })

          graph_selector.on("mouseover",function(d){
            d3.select(".graph_styles")
            .style("cursor","pointer")
          })


          //optimization of dataset - has been removed after first review
          /*var x;
          for(x in data){
          	data[x]["LoanOriginationDateAdjusted"] = String(data[x]["LoanOriginationDate"]).slice(5,7) +"-"+ String(data[x]["LoanOriginationDate"]).slice(0,4)
          }*/

          last_button = "IncomeRange";

          last_style = "Absolute";
          
          draw(data,last_button,last_style,svg);

          
                     };//

