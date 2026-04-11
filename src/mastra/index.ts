import { Mastra } from '@mastra/core'
import { weatherAgent } from './agents/weather-agent'
import { MyPrivateGateway, PRIVATE_GATEWAY_ID } from '../utils/cutom-gateway'

export const mastra = new Mastra({
    gateways: {
        [`${PRIVATE_GATEWAY_ID}Gateway`]: new MyPrivateGateway(),
    },
    agents: { weatherAgent },
})

mastra.addGateway(new MyPrivateGateway(), PRIVATE_GATEWAY_ID);