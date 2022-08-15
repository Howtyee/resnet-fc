// logs.js
const util = require('../../utils/util.js')

Page({
  data: {
    imgUrls: [
      {
        link: '/pages/logs/logs',
        url: '/pages/static/table/t1.1.png'

      }, {

        link: '/pages/logs/logs',
        url: '/pages/static/table/t1.2.png'
      }, {

        link: '/pages/logs/logs',
        url: '/pages/static/table/t1.3.png'


      },
      {
        link: '/pages/logs/logs',
        url: '/pages/static/table/t1.4.png'
      },
      {

        link: '/pages/logs/logs',
        url: '/pages/static/table/t1.5.png'
      }

    ],

    indicatorDots: true,  //小点

    autoplay: true,  //是否自动轮播

    interval: 3000,  //间隔时间

    duration: 3000,  //滑动时间
    

    logs: []
  },
  bindButtonTap:function(){
    var that = this;
      wx.chooseVideo({
      sourceType:['album','camera'],
      maxDuration:60,
      camera:['front','back'],
      success:function(res){
        that.setData({
          src:res.tempFilePath
        })
        console.log(res.tempFilePath);
      }
    })
  },
  onLoad() {
    this.setData({
      logs: (wx.getStorageSync('logs') || []).map(log => {
        return {
          date: util.formatTime(new Date(log)),
          timeStamp: log
        }
      })
    })
  }
})

