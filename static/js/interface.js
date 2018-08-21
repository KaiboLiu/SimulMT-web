function update_cpu_load() {
    
    //$.getJSON($SCRIPT_ROOT+"/_stuff",
    $.getJSON("/cpu",
        function(data) {
            $("#cpu").text(data.cpu+" %");
            $("#ram").text(data.ram+" % / "+(''+(parseFloat(data.ram)*0.16)).slice(0,5) + " G");
            $("#time").text(data.time) ;
        });
}

//console.log('in js');


// update the stream result in screen
function fillScreen() {
    //$.getJSON($SCRIPT_ROOT+"/_stuff",
    $.getJSON("/fillScreen",
        function(data) {
            var chnBox = $("#chnBox");
            var engBox = $("#chnBox");
            var bslBox = $("#baselineBox");
            chnBox.text(data.BoxDisplay_chn);
            engBox.text(data.BoxDisplay_eng);
            bslBox.text(data.BoxDisplay_baseline);
            if (data.scroll_chn)      { chnBox.animate({scrollTop:chnBox.scrollTop() + 21});  }
            if (data.scroll_eng)      { engBox.animate({scrollTop:engBox.scrollTop() + 18});  }
            if (data.scroll_baseline) { bslBox.animate({scrollTop:bslBox.scrollTop() + 18});  }
        });
}


setInterval(update_cpu_load, 1000);
setInterval(fillScreen, 100);






function doScroll(){
   //$('#engBox').animate({scrollTop:$('#engBox').scrollTop() + 20},400);

	 var thisdiv = $('#chnBox');
   var height = parseInt(document.getElementById('chnBox').style.lineHeight);

   //$('#engBox').animate({scrollTop:$('#engBox').scrollTop() + 20});
   
   var engdiv = $('#engBox');
   if (engdiv.scrollTop() > 800){
   		engdiv.animate({scrollTop:0});
   }else{
   		engdiv.animate({scrollTop:engdiv.scrollTop() + 15});
   }
   
   if (thisdiv.scrollTop() > 800){
   		thisdiv.animate({scrollTop:0});
   }else{
   		thisdiv.animate({scrollTop:thisdiv.scrollTop() + 20});
   }
   $('#link').text(engdiv.scrollTop() + '/' + thisdiv.scrollTop());
   //$('#link').innerHTML = $('#engBox').scrollTop() + '/' + $('#engBox').offsetHeight;
}

setInterval(doScroll, 1000);

